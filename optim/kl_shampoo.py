from itertools import chain

import torch


class KLShampoo(torch.optim.Optimizer):
    """KL-Shampoo with AdamW fallback for parameters that are unsafe to precondition.

    This is a small local implementation adapted for this repo. Matrix-like
    tensors with every dimension <= max_precondition_dim use KL-Shampoo.
    Vectors, scalars, and very large tensors use AdamW fallback.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.98),
        shampoo_beta=-1.0,
        eps=1e-8,
        weight_decay=0.01,
        precondition_frequency=10,
        normalize_grads=False,
        init_factor=0.1,
        max_precondition_dim=2048,
        cast_dtype=torch.bfloat16,
        clamp_preconditioner=True,
        max_clamp_value=4000,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "shampoo_beta": shampoo_beta,
            "eps": eps,
            "weight_decay": weight_decay,
            "precondition_frequency": precondition_frequency,
            "normalize_grads": normalize_grads,
        }
        super().__init__(params, defaults)
        self.init_factor = init_factor
        self.max_precondition_dim = max_precondition_dim
        self.cast_dtype = cast_dtype
        self.clamp_preconditioner = clamp_preconditioner
        self.max_clamp_value = max_clamp_value

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.detach()
                state = self.state[param]

                if not self._can_precondition(grad):
                    self._adamw_step(param, grad, state, group)
                    continue

                grad = grad.to(dtype=self.cast_dtype)
                if "step" not in state:
                    state["step"] = 0

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)

                if "Q" not in state:
                    shampoo_beta = group["shampoo_beta"]
                    if shampoo_beta < 0:
                        shampoo_beta = group["betas"][1]
                    self._init_preconditioner(
                        grad,
                        state,
                        group["precondition_frequency"],
                        shampoo_beta,
                    )
                    self._update_preconditioner(grad, state)
                    continue

                state["step"] += 1
                update = self._klshampoo_update(
                    state,
                    grad,
                    beta1=group["betas"][0],
                    damping=group["eps"],
                )
                self._update_preconditioner(grad, state)

                if group["normalize_grads"]:
                    update = update / (1e-30 + update.pow(2).mean().sqrt())

                if group["weight_decay"] > 0:
                    param.add_(param, alpha=-group["lr"] * group["weight_decay"])
                param.add_(update.view_as(param), alpha=-group["lr"])

        return loss

    def _can_precondition(self, grad):
        return grad.dim() in (2, 3) and max(grad.shape) <= self.max_precondition_dim

    def _adamw_step(self, param, grad, state, group):
        beta1, beta2 = group["betas"]
        if "adam_step" not in state:
            state["adam_step"] = 0
            state["adam_exp_avg"] = torch.zeros_like(param)
            state["adam_exp_avg_sq"] = torch.zeros_like(param)

        state["adam_step"] += 1
        exp_avg = state["adam_exp_avg"]
        exp_avg_sq = state["adam_exp_avg_sq"]

        exp_avg.lerp_(grad, 1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        bias_correction1 = 1.0 - beta1 ** state["adam_step"]
        bias_correction2 = 1.0 - beta2 ** state["adam_step"]
        step_size = group["lr"] * (bias_correction2**0.5) / bias_correction1

        if group["weight_decay"] > 0:
            param.add_(param, alpha=-group["lr"] * group["weight_decay"])
        param.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(group["eps"]), value=-step_size)

    def _init_preconditioner(self, grad, state, precondition_frequency, shampoo_beta):
        state["GG"] = [
            torch.zeros(size, size, device=grad.device, dtype=grad.dtype)
            for size in grad.shape
        ]
        state["Q"] = None
        state["precondition_frequency"] = precondition_frequency
        state["shampoo_beta"] = shampoo_beta

    def _update_s(self, grad, state, mat, idx, beta, total_factor):
        factor = total_factor / grad.shape[idx]
        state["GG"][idx].mul_(beta).add_(mat, alpha=(1.0 - beta) / factor)

    def _update_eigen_value(self, state, diag, idx, beta):
        inv_d = state["eigen_sqrt_inv"][idx] ** 2
        eig = (1.0 / inv_d).nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        eig.lerp_(diag, 1.0 - beta)
        sqrt_inv = (1.0 / torch.sqrt(eig)).nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        if self.clamp_preconditioner:
            clamp = max(10, min(sqrt_inv.shape[0], self.max_clamp_value))
            sqrt_inv = torch.clamp(sqrt_inv, max=clamp)
        state["eigen_sqrt_inv"][idx] = sqrt_inv

    def _update_preconditioner(self, grad, state):
        total_factor = torch.numel(grad)
        beta = state["shampoo_beta"]

        if state["Q"] is None:
            for idx in range(grad.dim()):
                mat = torch.tensordot(
                    grad,
                    grad,
                    dims=[[*chain(range(idx), range(idx + 1, grad.dim()))]] * 2,
                )
                self._update_s(grad, state, mat, idx, beta, total_factor)
            state["Q"], state["eigen_sqrt_inv"] = self._get_orthogonal_matrix(state["GG"])
        elif grad.dim() == 2:
            self._update_2d_preconditioner(grad, state, total_factor)
        else:
            self._update_3d_preconditioner(grad, state, total_factor)

        if state["step"] > 0 and state["step"] % state["precondition_frequency"] == 0:
            state["Q"] = self._get_orthogonal_matrix_qr(state)

    def _update_2d_preconditioner(self, grad, state, total_factor):
        beta = state["shampoo_beta"]

        step0 = state["Q"][1].T @ grad.T
        lhalf = step0 * state["eigen_sqrt_inv"][1].view(-1, 1)
        self._update_s(grad, state, lhalf.T @ lhalf, 0, beta, total_factor)

        step1 = state["Q"][0].T @ grad
        rhalf = step1 * state["eigen_sqrt_inv"][0].view(-1, 1)
        self._update_s(grad, state, rhalf.T @ rhalf, 1, beta, total_factor)

        diag_half = step1 @ state["Q"][1]
        ldiag = torch.mean((diag_half * state["eigen_sqrt_inv"][1].view(1, -1)) ** 2, dim=1)
        rdiag = torch.mean((diag_half * state["eigen_sqrt_inv"][0].view(-1, 1)) ** 2, dim=0)
        self._update_eigen_value(state, ldiag, 0, beta)
        self._update_eigen_value(state, rdiag, 1, beta)

    def _update_3d_preconditioner(self, grad, state, total_factor):
        beta = state["shampoo_beta"]
        inv_s = [
            q * eig.view(1, -1)
            for q, eig in zip(state["Q"], state["eigen_sqrt_inv"])
        ]

        g = grad
        g1 = torch.einsum("ija,ip->pja", g, inv_s[0])
        g1q2 = torch.einsum("pja,jl->pla", g1, state["Q"][1])
        g1q2q3 = torch.einsum("pqa,am->pqm", g1q2, state["Q"][2])

        g12 = g1q2 * state["eigen_sqrt_inv"][1].view(1, -1, 1)
        self._update_s(g, state, torch.tensordot(g12, g12, dims=[[0, 1], [0, 1]]), 2, beta, total_factor)

        g1q3 = torch.einsum("pqb,bm->pqm", g1, state["Q"][2])
        g13 = g1q3 * state["eigen_sqrt_inv"][2].view(1, 1, -1)
        self._update_s(g, state, torch.tensordot(g13, g13, dims=[[0, 2], [0, 2]]), 1, beta, total_factor)

        diag3 = torch.mean((g1q2q3 * state["eigen_sqrt_inv"][1].view(1, -1, 1)) ** 2, dim=(0, 1))
        diag2 = torch.mean((g1q2q3 * state["eigen_sqrt_inv"][2].view(1, 1, -1)) ** 2, dim=(0, 2))
        self._update_eigen_value(state, diag3, 2, beta)
        self._update_eigen_value(state, diag2, 1, beta)

        g3 = torch.einsum("ijb,bm->ijm", g, inv_s[2])
        g3q2 = torch.einsum("ijm,jq->iqm", g3, state["Q"][1])
        g3q2q1 = torch.einsum("iqm,ip->pqm", g3q2, state["Q"][0])
        g32 = g3q2 * state["eigen_sqrt_inv"][1].view(1, -1, 1)
        self._update_s(g, state, torch.tensordot(g32, g32, dims=[[1, 2], [1, 2]]), 0, beta, total_factor)

        diag1 = torch.mean((g3q2q1 * state["eigen_sqrt_inv"][1].view(1, -1, 1)) ** 2, dim=(1, 2))
        self._update_eigen_value(state, diag1, 0, beta)

    def _klshampoo_update(self, state, grad, beta1, damping):
        exp_avg = state["exp_avg"]
        exp_avg.lerp_(grad, 1.0 - beta1)

        projected = self._project(exp_avg, state)
        if grad.dim() == 2:
            inv = state["eigen_sqrt_inv"][0].view(-1, 1) * state["eigen_sqrt_inv"][1].view(1, -1)
        else:
            inv = (
                state["eigen_sqrt_inv"][0].view(-1, 1, 1)
                * state["eigen_sqrt_inv"][1].view(1, -1, 1)
                * state["eigen_sqrt_inv"][2].view(1, 1, -1)
            )

        inv = inv / (1.0 + inv * damping)
        return self._project_back(projected * inv, state)

    def _project(self, grad, state):
        for mat in state["Q"]:
            grad = torch.tensordot(grad, mat, dims=[[0], [0]])
        return grad

    def _project_back(self, grad, state):
        for mat in state["Q"]:
            grad = torch.tensordot(grad, mat, dims=[[0], [1]])
        return grad

    def _get_orthogonal_matrix(self, matrices):
        bases = []
        eig_inv = []
        for mat in matrices:
            original_dtype = mat.dtype
            original_device = mat.device
            mat32 = mat.float()
            eye = torch.eye(mat32.shape[0], device=mat32.device, dtype=mat32.dtype)
            try:
                _, q = torch.linalg.eigh(mat32 + 1e-30 * eye)
            except RuntimeError:
                _, q = torch.linalg.eigh((mat32 + 1e-30 * eye).double())
                q = q.float()

            q = torch.flip(q, dims=[1])
            inv = torch.ones(q.shape[0], device=q.device, dtype=q.dtype) * self.init_factor
            inv = (1.0 / torch.sqrt(inv)).nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            bases.append(q.to(device=original_device, dtype=original_dtype))
            eig_inv.append(inv.to(device=original_device, dtype=original_dtype))
        return bases, eig_inv

    def _get_orthogonal_matrix_qr(self, state):
        bases = []
        for mat, old_q in zip(state["GG"], state["Q"]):
            original_dtype = mat.dtype
            original_device = mat.device
            q, _ = torch.linalg.qr(mat.float() @ old_q.float())
            bases.append(q.to(device=original_device, dtype=original_dtype))
        return bases
