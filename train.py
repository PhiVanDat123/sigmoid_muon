import os
import math
import torch
from loguru import logger
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import (
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2Tokenizer,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm


class MoonDataset(Dataset):
    def __init__(self, dataset_name, dataset, tokenizer, max_length=512, split="train"):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.texts = dataset["text"]
        self.max_length = max_length
        self.split = split
        self.tokens = []
        self._tokenize_texts()

    def _tokenize_texts(self):
        cache_path = f"{self.dataset_name}_{self.split}.bin"
        if os.path.exists(cache_path):
            self.tokens = torch.load(cache_path)
        else:
            for text in tqdm(self.texts, desc=f"Tokenizing {self.split} texts"):
                encoded = self.tokenizer.encode(text, add_special_tokens=True)
                self.tokens.extend(encoded)
            torch.save(self.tokens, cache_path)

    def __len__(self):
        return len(self.tokens) // self.max_length

    def __getitem__(self, idx):
        start_idx = idx * self.max_length
        end_idx = start_idx + self.max_length
        token_slice = self.tokens[start_idx:end_idx]
        data = torch.tensor(token_slice, dtype=torch.long)
        return data


@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

@torch.compile
def newton_schulz_sigmoid_rect(G, iterations=5):
    squeezed = G.ndim == 1
    if squeezed:
        G = G.unsqueeze(0)

    m, n = G.shape
    # Dùng torch thay numpy để tính trên GPU
    norm_G = G.norm(p='fro')

    Q = G / (norm_G + 1e-8)
    In = torch.eye(n, device=G.device, dtype=G.dtype)  # ← torch, không phải np.eye

    for _ in range(iterations):
        Q = 0.5 * Q @ (3 * In - Q.T @ Q)

    T = G / 4.0
    for _ in range(2):
        T = 0.5 * T @ (3 * In - T.T @ T)

    result = 0.5 * Q + 0.5 * T

    if squeezed:
        result = result.squeeze(0)
    return result

class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    ):
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        for p in muon_params:
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
                p.data.mul_(1 - lr * wd)
                p.data.add_(u, alpha=-adjusted_lr)

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)
                g = buf1 / (eps + buf2.sqrt())
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss


class Muon_sigmoid(torch.optim.Optimizer):
    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    ):
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        for p in muon_params:
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                u = newton_schulz_sigmoid_rect(g, steps=group["ns_steps"])
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
                p.data.mul_(1 - lr * wd)
                p.data.add_(u, alpha=-adjusted_lr)

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)
                g = buf1 / (eps + buf2.sqrt())
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss


def get_model_and_dataloader(model_name, dataset_name, hidden_size, val_ratio=0.01):
    name2path = {
        "openwebtext-100k": "Elriggs/openwebtext-100k",
    }

    raw_dataset = load_dataset(name2path[dataset_name], trust_remote_code=True)

    # Chia train/val từ tập train gốc
    split = raw_dataset["train"].train_test_split(test_size=val_ratio, seed=42)
    train_split = split["train"]
    val_split = split["test"]

    if model_name == "qwen":
        tokenizer = Qwen2Tokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B", trust_remote_code=True
        )
    else:
        assert 0, f"model {model_name} not supported"

    train_dataset = MoonDataset(dataset_name, train_split, tokenizer, split="train")
    val_dataset = MoonDataset(dataset_name, val_split, tokenizer, split="val")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    if model_name == "qwen":
        config = Qwen2Config(
            attention_dropout=0.0,
            bos_token_id=151643,
            eos_token_id=151643,
            hidden_act="silu",
            hidden_size=hidden_size,
            initializer_range=0.02,
            intermediate_size=4864,
            max_position_embeddings=513,
            max_window_layers=12,
            model_type="qwen2",
            num_attention_heads=16,
            num_hidden_layers=12,
            num_key_value_heads=16,
            rms_norm_eps=1e-06,
            rope_theta=1000000.0,
            sliding_window=1024,
            tie_word_embeddings=True,
            torch_dtype="bfloat16",
            use_cache=True,
            use_mrope=False,
            use_sliding_window=False,
            vocab_size=151936,
        )
        model = Qwen2ForCausalLM(config)
    else:
        assert 0, f"model {model_name} not supported"

    return model, train_loader, val_loader


def get_optimizer(optimizer_name, model, lr=1e-3, wd=0.1):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95)
        )
    elif optimizer_name == "muon":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
            )
        ]
        return Muon(
            lr=lr,
            wd=wd,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )
    elif optimizer_name == "muon_sigmoid":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
            )
        ]
        return Muon_sigmoid(
            lr=lr,
            wd=wd,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )
    else:
        assert 0, "optimizer not supported"


@torch.no_grad()
def evaluate(model, val_loader, device, num_batches=50):
    """Tính val loss trên num_batches batch, không cần chạy hết toàn bộ val set."""
    model.eval()
    total_loss = 0.0
    count = 0
    for i, batch in enumerate(val_loader):
        if i >= num_batches:
            break
        batch = batch.to(device)
        outputs = model(input_ids=batch, labels=batch)
        total_loss += outputs.loss.item()
        count += 1
    model.train()
    return total_loss / count if count > 0 else float("inf")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen")
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, default="openwebtext-100k")
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--val_every", type=int, default=3000,
                        help="Evaluate val loss every N steps")
    parser.add_argument("--val_ratio", type=float, default=0.01,
                        help="Fraction of data dùng làm validation (default: 5%%)")
    args = parser.parse_args()
    logger.add(f"logs/train_{args.model}_{args.optimizer}_lr{args.lr}.log")

    model, train_loader, val_loader = get_model_and_dataloader(
        args.model, args.dataset, args.hidden_size, val_ratio=args.val_ratio
    )
    optimizer = get_optimizer(args.optimizer, model, lr=args.lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    num_epochs = 1
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_loader) * num_epochs,
        num_cycles=0.5,
    )

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_loader):
            batch = batch.to(device)
            outputs = model(input_ids=batch, labels=batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            logger.info(
                f"Epoch: {epoch} Step: {step} LR: {optimizer.param_groups[0]['lr']:.6f} "
                f"Train loss: {loss.item():.4f}"
            )

            # Evaluate val loss mỗi val_every steps
            if (step + 1) % args.val_every == 0:
                val_loss = evaluate(model, val_loader, device, num_batches=50)
                logger.info(
                    f">>> [VAL] Epoch: {epoch} Step: {step + 1} "
                    f"Val loss: {val_loss:.4f} | Val perplexity: {math.exp(val_loss):.2f}"
                )