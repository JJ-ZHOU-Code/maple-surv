"""
SCPP: Subtype-Conditional Prognostic Prompts
Joint survival + subtype classification with stopgrad to prevent subtype shortcuts.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vlsa import VLSA


class SCPP(VLSA):
    """
    Extends VLSA with:
      1. subtype_head: linear classifier on top of MIL features
      2. subtype_surv_residuals: per-subtype residuals added to survival text features
      3. stopgrad: detach subtype probs before conditioning survival path

    Forward returns (surv_logits, img_feat, text_feat_cond, subtype_logits).
    Training uses joint loss: L_surv + lambda * L_cls.
    """

    def __init__(
        self,
        text_encoder_cfg,
        image_encoder_cfg,
        prompt_learner_cfg,
        pretrained_prompt_learner_cfg=None,
        num_subtypes: int = 2,
        feat_dim: int = 512,
        stopgrad: bool = True,
        routing: str = "soft",
        info_prefix: str = "SCPP",
        **kwargs,
    ):
        super().__init__(
            text_encoder_cfg,
            image_encoder_cfg,
            prompt_learner_cfg,
            pretrained_prompt_learner_cfg,
            info_prefix=info_prefix,
            **kwargs,
        )
        assert routing in ("soft", "hard", "oracle"), f"Unknown routing: {routing}"

        self.num_subtypes = num_subtypes
        self.stopgrad = stopgrad
        self.routing = routing

        # Subtype classification head
        self.subtype_head = nn.Linear(feat_dim, num_subtypes)

        # Per-subtype survival text-feature residuals [K, R, feat_dim]
        # num_ranks determined at runtime from text features; use lazy init
        self._feat_dim = feat_dim
        self._residuals_initialized = False

    def _maybe_init_residuals(self, num_ranks: int, device):
        """Lazily initialise residuals once we know num_ranks."""
        if not self._residuals_initialized:
            self.subtype_surv_residuals = nn.Parameter(
                torch.zeros(self.num_subtypes, num_ranks, self._feat_dim, device=device)
            )
            self._residuals_initialized = True

    def _compute_subtype_routing(self, subtype_logits, oracle_labels=None):
        """Return subtype probability weights [B, K]."""
        if self.routing == "oracle" and oracle_labels is not None:
            # One-hot from ground-truth labels (upper bound)
            B, K = subtype_logits.shape
            p = torch.zeros(B, K, device=subtype_logits.device)
            p.scatter_(1, oracle_labels.view(-1, 1), 1.0)
            return p
        elif self.routing == "hard":
            # Gumbel-softmax (straight-through) during training, argmax at test
            if self.training:
                return F.gumbel_softmax(subtype_logits, tau=1.0, hard=True)
            else:
                idx = subtype_logits.argmax(dim=-1, keepdim=True)
                p = torch.zeros_like(subtype_logits).scatter_(1, idx, 1.0)
                return p
        else:
            # Default: soft routing
            return F.softmax(subtype_logits, dim=-1)

    def forward(self, X, oracle_subtype=None):
        """
        X: [1, N, feat_dim] — bag of instance features
        oracle_subtype: optional int tensor [1] for oracle routing (GT subtype label)
        Returns (surv_logits, img_feat, text_feat_cond, subtype_logits)
        """
        # ── 1. Survival text features (base) ─────────────────────────────────
        text_features = self.forward_text_only()            # [R, 512]
        text_features = F.normalize(text_features, dim=-1)

        # ── 2. Image features from MIL encoder ───────────────────────────────
        image_features = self.encode_instances(X)           # [1, 512]
        image_features = F.normalize(image_features, dim=-1)

        # ── 3. Subtype classification ─────────────────────────────────────────
        subtype_logits = self.subtype_head(image_features)  # [1, K]
        subtype_probs  = self._compute_subtype_routing(subtype_logits, oracle_subtype)  # [1, K]

        # ── 4. Subtype-conditional text feature residuals ─────────────────────
        R = text_features.shape[0]
        self._maybe_init_residuals(R, text_features.device)

        p_cond = subtype_probs.detach() if self.stopgrad else subtype_probs  # [1, K]
        # Weighted sum of per-subtype residuals: [R, 512]
        cond_residuals = torch.einsum("bk,krd->rd", p_cond, self.subtype_surv_residuals)

        text_features_cond = F.normalize(text_features + cond_residuals, dim=-1)

        # ── 5. Survival logits ────────────────────────────────────────────────
        logit_scale  = self.logit_scale.exp()
        surv_logits  = logit_scale * image_features @ text_features_cond.t()  # [1, R]

        return surv_logits, image_features, text_features_cond, subtype_logits
