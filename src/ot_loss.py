"""
OT-LoRA: decoder-side Optimal Transport supervision for medical MLLM report
generation under parameter-efficient fine-tuning.

Framework-agnostic loss module. Works with any model that can expose
  (a) vision patch features  [B, Nv, Dv]   (vision tower output)
  (b) decoder hidden states  [B, Nt, Dt]   (LLM layers, report tokens only)
  (c) per-token salience     [B, Nt]       (non-negative; entity tokens upweighted)

Deltas w.r.t. LOTUS (AACL 2026 Findings, compact R2Gen encoder-decoder):
  1. Salience-weighted token marginals  (uniform -> clinically weighted)
  2. Transport-mass gating ("partial" transport): low-salience template tokens
     may be excluded from the transport problem entirely (hard gate) or
     down-weighted (soft), instead of being forced to carry mass.
  3. Same entropic Sinkhorn core + ratio-capped weighting, proven stable.

Sinkhorn core is ported from r2-gen-ot/modules/ot_utils.py (log-domain solver,
default) and kept differentiable w.r.t. both point sets and the marginals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Cost functions
# ---------------------------------------------------------------------------

def pairwise_squared_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x: [B, Nx, D], y: [B, Ny, D] -> [B, Nx, Ny]"""
    x_norm = (x ** 2).sum(dim=-1, keepdim=True)
    y_norm = (y ** 2).sum(dim=-1, keepdim=True).transpose(1, 2)
    dist = x_norm + y_norm - 2.0 * torch.bmm(x, y.transpose(1, 2))
    return torch.clamp(dist, min=0.0)


def pairwise_mean_squared_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return pairwise_squared_distance(x, y) / float(max(1, x.size(-1)))


def pairwise_cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return 1.0 - torch.bmm(x, y.transpose(1, 2))


def pairwise_distance(x, y, cost_type="l2_mean"):
    if cost_type == "cosine":
        return pairwise_cosine_distance(x, y)
    if cost_type == "l2":
        return pairwise_squared_distance(x, y)
    return pairwise_mean_squared_distance(x, y)


# ---------------------------------------------------------------------------
# Sinkhorn (log-domain, differentiable w.r.t. costs and marginals)
# ---------------------------------------------------------------------------

def sinkhorn_log(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    valid: torch.Tensor,
    epsilon: float = 0.1,
    n_iters: int = 30,
    cost_type: str = "l2_mean",
    unbalanced_tau: float = 0.8,
    balanced: bool = True,
):
    """
    Entropic OT between two weighted point sets.

    x: [B, Nx, D]; y: [B, Ny, D]
    a: [B, Nx] source marginal (non-negative, will be normalized inside)
    b: [B, Ny] target marginal
    valid: [B, Nx, Ny] boolean mask of allowed couplings
    balanced=False activates the tau-relaxed (unbalanced) updates, letting the
    plan leak mass where marginals are soft (used for template tokens).

    Returns (transport_plan [B, Nx, Ny], cost_scalar [B]).
    """
    a = a / (a.sum(dim=1, keepdim=True) + 1e-8)
    b = b / (b.sum(dim=1, keepdim=True) + 1e-8)

    cost = pairwise_distance(x, y, cost_type=cost_type)
    logK = (-cost / epsilon).masked_fill(~valid, -1e9)
    log_a = torch.log(a.clamp_min(1e-32))
    log_b = torch.log(b.clamp_min(1e-32))

    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    tau = 1.0 if balanced else unbalanced_tau

    for _ in range(n_iters):
        logKv = torch.logsumexp(logK + log_v.unsqueeze(1), dim=2)
        log_u = tau * (log_a - logKv)
        logKTu = torch.logsumexp(logK.transpose(1, 2) + log_u.unsqueeze(1), dim=2)
        log_v = tau * (log_b - logKTu)

    logT = log_u.unsqueeze(2) + logK + log_v.unsqueeze(1)
    plan = torch.exp(logT) * valid.to(x.dtype)
    cost_scalar = (plan * cost * valid.to(x.dtype)).sum(dim=(1, 2))
    return plan, cost_scalar


# ---------------------------------------------------------------------------
# Salience-weighted CA-OT loss
# ---------------------------------------------------------------------------

class CAOTLoss(nn.Module):
    """
    Total objective:
      L = lam_local * Sinkhorn(V, T; a, b)
        + lam_global * || bary(V; a) - bary(T; b) ||_2^2

    Marginal design (key ablation axis):
      token_marginal:
        'uniform'   b_j = 1/Nt                        (LOTUS behaviour)
        'salience'  b_j ∝ w_j (entity-weighted)       (soft)
        'gated'     only tokens with w_j >= gate_thr   (hard partial transport)
      patch_marginal:
        'uniform'   a_i = 1/Nv
        'salience'  a_i ∝ v_j (e.g. foreground mass)

    Projection: both modalities are mapped to a shared low-dim space and
    L2-normalized before transport (d_proj=256, following LOTUS).
    """

    def __init__(
        self,
        d_vision: int,
        d_text: int,
        d_proj: int = 256,
        lam_local: float = 0.7,
        lam_global: float = 0.3,
        epsilon: float = 0.1,
        n_iters: int = 30,
        cost_type: str = "l2_mean",
        token_marginal: str = "salience",   # 'uniform' | 'salience' | 'gated'
        patch_marginal: str = "uniform",
        gate_threshold: float = 0.5,
        balanced: bool = True,
        unbalanced_tau: float = 0.8,
        proj_mode: str = "fixed",  # 'fixed' | 'fixed_shared' | 'learned'
    ):
        super().__init__()
        assert abs(lam_local + lam_global - 1.0) < 1e-6, "lambdas must sum to 1"
        self.proj_mode = proj_mode
        if proj_mode == "learned":
            self.v_proj = nn.Linear(d_vision, d_proj)
            self.t_proj = nn.Linear(d_text, d_proj)
        else:
            # Fixed random JL projection: distance-preserving, NOT trainable.
            # Rationale: with learned projections under PEFT, both heads can
            # collapse to a constant map, driving transport cost to zero with
            # no grounding (observed empirically: OT loss -> 1e-4 and all
            # marginal variants become identical).
            # 'fixed_shared': one matrix for BOTH modalities. With independent
            # matrices the cost is a random bilinear form v'(P_v P_t')h whose
            # expectation is 0 for ANY pair -- JL only preserves distances
            # within one projected space. A shared P makes ||Pv - Ph|| track
            # the genuine overlap of v and h in the raw space.
            W = torch.randn(d_vision, d_proj) / (d_vision ** 0.5)
            self.register_buffer("v_W", W)
            if proj_mode == "fixed_shared":
                assert d_vision == d_text, "shared projection needs matching dims"
                self.register_buffer("t_W", self.v_W)  # same storage: one shared matrix
            else:
                self.register_buffer("t_W", torch.randn(d_text, d_proj) / (d_text ** 0.5))
        self.lam_local, self.lam_global = lam_local, lam_global
        self.epsilon, self.n_iters, self.cost_type = epsilon, n_iters, cost_type
        self.token_marginal = token_marginal
        self.patch_marginal = patch_marginal
        self.gate_threshold = gate_threshold
        self.balanced = balanced
        self.unbalanced_tau = unbalanced_tau

    def forward(
        self,
        vision_feats: torch.Tensor,      # [B, Nv, Dv] patch features
        text_hiddens: torch.Tensor,      # [B, Nt, Dt] decoder hidden states (report tokens)
        token_salience: torch.Tensor = None,  # [B, Nt] non-negative weights
        patch_salience: torch.Tensor = None,  # [B, Nv]
        text_mask: torch.Tensor = None,       # [B, Nt] 1 for real report tokens
    ):
        B, Nv, _ = vision_feats.shape
        Nt = text_hiddens.size(1)
        # Sinkhorn in fp32 even under bf16 autocast (logsumexp stability)
        vision_feats = vision_feats.float()
        text_hiddens = text_hiddens.float()
        dev, dt = vision_feats.device, vision_feats.dtype

        if text_mask is None:
            text_mask = torch.ones(B, Nt, device=dev, dtype=dt)
        if token_salience is None:
            token_salience = torch.ones(B, Nt, device=dev, dtype=dt)
        text_mask = text_mask.to(dt)
        token_salience = token_salience.to(dt) * text_mask  # salience only on real tokens

        if self.proj_mode == "learned":
            v = F.normalize(self.v_proj(vision_feats), dim=-1)
            t = F.normalize(self.t_proj(text_hiddens), dim=-1)
        else:
            v = F.normalize(torch.matmul(vision_feats, self.v_W), dim=-1)
            t = F.normalize(torch.matmul(text_hiddens, self.t_W), dim=-1)

        # ----- marginals -----
        if self.token_marginal == "uniform":
            b = text_mask.clone()
        elif self.token_marginal == "salience":
            b = token_salience.clamp_min(0.0)
        elif self.token_marginal == "gated":
            b = (token_salience >= self.gate_threshold).to(dt) * text_mask
            b = b + 1e-3 * text_mask  # keep tiny mass everywhere for numerical safety
        else:
            raise ValueError(self.token_marginal)

        if self.patch_marginal == "uniform" or patch_salience is None:
            a = torch.ones(B, Nv, device=dev, dtype=dt)
        else:
            a = patch_salience.to(dt).clamp_min(1e-6)

        valid = (a.unsqueeze(2) > 0) & (b.unsqueeze(1) > 0)

        plan, local_cost = sinkhorn_log(
            v, t, a, b, valid,
            epsilon=self.epsilon, n_iters=self.n_iters, cost_type=self.cost_type,
            balanced=self.balanced, unbalanced_tau=self.unbalanced_tau,
        )

        # ----- global barycenter consistency -----
        a_n = a / (a.sum(dim=1, keepdim=True) + 1e-8)
        b_n = b / (b.sum(dim=1, keepdim=True) + 1e-8)
        v_bar = torch.bmm(a_n.unsqueeze(1), v).squeeze(1)   # [B, d_proj]
        t_bar = torch.bmm(b_n.unsqueeze(1), t).squeeze(1)
        global_loss = ((v_bar - t_bar) ** 2).sum(dim=-1)

        loss = self.lam_local * local_cost + self.lam_global * global_loss
        return loss.mean(), {"plan": plan, "local": local_cost.mean().item(),
                             "global": global_loss.mean().item()}


# ---------------------------------------------------------------------------
# Ratio-capped auxiliary weighting (LOTUS Eq.12, stability mechanism)
# ---------------------------------------------------------------------------

def ratio_capped_weight(aux_loss: torch.Tensor, lm_loss: torch.Tensor,
                        nominal_w: float, rho: float, delta: float = 1e-6) -> torch.Tensor:
    """Effective OT weight w~ = min(nominal_w, rho * sg(L_lm) / (L_aux + delta)),
    recomputed per step so the OT branch never exceeds rho times the LM loss."""
    ratio = rho * lm_loss.detach() / (aux_loss.detach() + delta)
    return torch.clamp(ratio, max=nominal_w)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, Nv, Nt, Dv, Dt = 4, 49, 60, 1280, 2048
    vis = torch.randn(B, Nv, Dv)
    txt = torch.randn(B, Nt, Dt, requires_grad=True)
    sal = torch.rand(B, Nt) * 2.0
    mask = torch.ones(B, Nt)

    for tm in ["uniform", "salience", "gated"]:
        crit = CAOTLoss(Dv, Dt, token_marginal=tm)
        loss, info = crit(vis, txt, sal, None, mask)
        loss.backward()
        assert torch.isfinite(loss), tm
        assert txt.grad is not None and torch.isfinite(txt.grad).all(), tm
        print(f"token_marginal={tm:9s} loss={loss.item():.4f} "
              f"plan_sum={info['plan'].sum(dim=(1,2)).mean().item():.4f}")
    print("ot_loss self-test OK")
