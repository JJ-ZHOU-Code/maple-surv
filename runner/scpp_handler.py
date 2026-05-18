"""
SCPPHandler: training handler for joint survival + subtype classification.
Extends VLSAHandler; adds:
  - subtype label extraction from the 3rd column of the label tensor
  - joint loss = L_surv + cls_weight * L_cls
  - subtype accuracy logging
"""
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from functools import partial

from .vlsa_handler import VLSAHandler
from model.utils import load_model, general_init_weight
from utils.func import fetch_kws, freeze_param


class SCPPHandler(VLSAHandler):
    """
    Handler for SCPP (Subtype-Conditional Prognostic Prompts).

    Config additions over VLSAHandler:
      scpp_num_subtypes     : int  (default 2)
      scpp_stopgrad         : bool (default True)
      scpp_routing          : str  'soft'|'hard'|'oracle' (default 'soft')
      scpp_feat_dim         : int  (default 512)
      loss_cls_weight       : float (weight for classification cross-entropy, default 1.0)
    """

    def __init__(self, cfg):
        # Override task check -- SCPPHandler accepts task='scpp'
        assert cfg['task'] == 'scpp', f"Expected task='scpp' but got {cfg['task']}."
        # Temporarily patch task for parent init (parent checks task='vlsa')
        cfg['task'] = 'vlsa'
        super().__init__(cfg)
        cfg['task'] = 'scpp'
        self.cfg = cfg

    @staticmethod
    def func_load_meta_data(cfg, data_split=None):
        """Extends VLSAHandler loader to add 'subtype' to the label columns."""
        meta_data = VLSAHandler.func_load_meta_data(cfg, data_split=data_split)
        # Append 'subtype' to column_label if the CSV has a 'subtype' column
        if 'subtype' in meta_data.pat_data.columns:
            meta_data.column_label = meta_data.column_label + ['subtype']
            print("[SCPPHandler] Added 'subtype' to label columns.")
        else:
            print("[SCPPHandler] Warning: 'subtype' column not found in survival CSV.")
        return meta_data

    @staticmethod
    def func_load_model(cfg):
        """Load SCPP model instead of VLSA."""
        arch = cfg['arch']
        assert arch == 'SCPP', f"Expected arch='SCPP' but got {arch}."

        pmt_learner_name = cfg['vlsa_pmt_learner_name']
        prompt_learner_cfg = fetch_kws(cfg, prefix='vlsa_pmt_learner_' + pmt_learner_name.lower())
        prompt_learner_cfg.update({"name": pmt_learner_name})

        pmt_learner_pretrained = cfg.get('vlsa_pmt_learner_pretrained', False)
        prompt_learner_cfg['pretrained'] = pmt_learner_pretrained
        pretrained_prompt_learner_cfg = None
        if pmt_learner_pretrained:
            pretrained_prompt_learner_cfg = fetch_kws(cfg, prefix='vlsa_pmt_learner_coop')
            pretrained_prompt_learner_cfg['ckpt'] = pretrained_prompt_learner_cfg['ckpt'].format(
                cfg['data_split_seed'], pretrained_prompt_learner_cfg['method']
            )

        text_encoder_cfg  = fetch_kws(cfg, prefix='vlsa_txt_encoder')
        image_encoder_cfg = fetch_kws(cfg, prefix='vlsa_img_encoder')

        arch_cfg = {
            'vlsa_api':  cfg['vlsa_api'],
            'text_encoder_cfg':   text_encoder_cfg,
            'image_encoder_cfg':  image_encoder_cfg,
            'prompt_learner_cfg': prompt_learner_cfg,
            'pretrained_prompt_learner_cfg': pretrained_prompt_learner_cfg,
            'path_clip_model': cfg['path_clip_model'],
            'num_subtypes': cfg.get('scpp_num_subtypes', 2),
            'feat_dim':     cfg.get('scpp_feat_dim', 512),
            'stopgrad':     cfg.get('scpp_stopgrad', True),
            'routing':      cfg.get('scpp_routing', 'soft'),
        }
        model = load_model('SCPP', **arch_cfg)

        if cfg.get('init_wt', False):
            model.apply(general_init_weight)

        cfg_frozen = [
            ('mil_encoder',  model.mil_encoder,  image_encoder_cfg['frozen']),
            ('text_encoder', model.prompt_encoder, text_encoder_cfg['frozen']),
            ('logit_scale',  model.logit_scale,   cfg.get('vlsa_frozen_logit_scale', False)),
        ]
        for name, module, freeze_it in cfg_frozen:
            if freeze_it:
                print(f"[SCPPHandler] Freezing {name}.")
                try:
                    freeze_param(module)
                except AttributeError:
                    pass

        return model

    def _check_arguments(self, cfg):
        # Skip parent checks (they enforce task='vlsa')
        pass

    def _train_each_epoch(self, epoch, train_loader, name_loader):
        self.net.train()
        bp_every_batch = self.cfg['bp_every_batch']
        all_raw_pred, all_gt, all_idx = [], [], []

        idx_collector, x_collector, y_collector = [], [], []
        i_batch = 0
        num_samples = len(train_loader)
        loop = tqdm(train_loader, desc=name_loader)
        for data_idx, data_x, data_y in loop:
            i_batch += 1
            data_input = data_x[0].cuda()
            data_label = data_y.cuda()

            x_collector.append(data_input)
            y_collector.append(data_label)
            idx_collector.append(data_idx)

            if i_batch % bp_every_batch == 0 or i_batch == num_samples:
                batch_loss, batch_pred = self._update_network(x_collector, y_collector)
                all_raw_pred.append(batch_pred)
                all_gt.append(torch.cat(y_collector, dim=0)[:, :2].detach().cpu())
                all_idx.append(torch.cat(idx_collector, dim=0).detach().cpu())

                idx_collector, x_collector, y_collector = [], [], []
                torch.cuda.empty_cache()

                wandb.log({'train/batch_loss': batch_loss})
                loop.set_description(f"Epoch [{epoch}/{self.cfg['epochs']}]")
                loop.set_postfix(loss=batch_loss)

        all_raw_pred = torch.cat(all_raw_pred, dim=0)
        all_gt       = torch.cat(all_gt, dim=0)
        all_idx      = torch.cat(all_idx, dim=0).squeeze(-1)

        train_cltor = dict()
        all_pred    = self.output_converter(all_raw_pred)
        all_uids    = self._get_unique_id('train', all_idx)
        train_cltor['pred'] = {'y': all_gt, 'raw_y_hat': all_raw_pred, 'y_hat': all_pred, 'uid': all_uids}
        return train_cltor

    def _update_network(self, xs, ys):
        n_sample = len(xs)
        surv_preds, cls_preds, surv_labels, cls_labels = [], [], [], []

        for i in range(n_sample):
            label = ys[i]  # [1, 3] or [1, 2]
            has_subtype = label.shape[-1] >= 3

            surv_logits, _, _, sub_logits = self.net(xs[i])
            surv_preds.append(surv_logits)
            cls_preds.append(sub_logits)
            surv_labels.append(label[:, :2])
            if has_subtype:
                cls_labels.append(label[:, 2].long())

        self.optimizer.zero_grad()

        bag_surv_preds = torch.cat(surv_preds, dim=0)
        bag_surv_label = torch.cat(surv_labels, dim=0)
        surv_loss = self.calc_objective_loss(bag_surv_preds, bag_surv_label)

        total_loss = surv_loss
        if cls_labels:
            bag_cls_preds  = torch.cat(cls_preds, dim=0)
            bag_cls_labels = torch.cat(cls_labels, dim=0)
            cls_loss   = F.cross_entropy(bag_cls_preds, bag_cls_labels)
            cls_weight = self.cfg.get('loss_cls_weight', 1.0)
            total_loss = surv_loss + cls_weight * cls_loss
            wandb.log({'train/cls_loss': cls_loss.item()})

        if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
            total_loss.backward()
            self.optimizer.step()
            val_loss = total_loss.item()
        else:
            print("[batch train] warning: loss not evaluated; skipped.")
            val_loss = 0

        return val_loss, bag_surv_preds.detach().cpu()

    def test_model(self, model, loader, loader_name, ckpt_path=None):
        if ckpt_path is not None:
            net_ckpt = torch.load(ckpt_path)
            model.load_state_dict(net_ckpt['model'], strict=False)
        model.eval()

        all_idx, all_raw_pred, all_pred, all_gt = [], [], [], []
        cls_correct, cls_total = 0, 0

        for data_idx, data_x, data_y in loader:
            X = data_x[0].cuda()
            with torch.no_grad():
                surv_logits, _, _, sub_logits = model(X)
                pred = self.output_converter(surv_logits)

            label = data_y
            all_gt.append(label[:, :2])
            all_raw_pred.append(surv_logits.detach().cpu())
            all_pred.append(pred.detach().cpu())
            all_idx.append(data_idx)

            if label.shape[-1] >= 3:
                gt_sub   = label[:, 2].long()
                pred_sub = sub_logits.argmax(dim=-1).cpu()
                cls_correct += (pred_sub == gt_sub).sum().item()
                cls_total   += gt_sub.shape[0]

        all_raw_pred = torch.cat(all_raw_pred, dim=0)
        all_pred     = torch.cat(all_pred, dim=0)
        all_gt       = torch.cat(all_gt, dim=0)
        all_idx      = torch.cat(all_idx, dim=0).squeeze()

        if cls_total > 0:
            acc = cls_correct / cls_total
            print(f"[{loader_name}] Subtype accuracy: {acc:.4f} ({cls_correct}/{cls_total})")
            wandb.log({f'{loader_name}/subtype_acc': acc})

        cltor = dict()
        all_uids = self._get_unique_id(loader_name, all_idx)
        cltor['pred'] = {'y': all_gt, 'raw_y_hat': all_raw_pred, 'y_hat': all_pred, 'uid': all_uids}
        return cltor
