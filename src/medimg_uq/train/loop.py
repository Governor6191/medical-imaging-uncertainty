"""The training loop and ensemble orchestration.

Config-driven and task-agnostic: it moves batches to the device, runs mixed
precision when on CUDA, validates each epoch, and keeps the best checkpoint by
validation loss. The loss comes from the task tag, the data from a ``DataLoader``
of ``Batch`` objects, so the same loop trains an ISIC classifier or a BraTS
segmenter without change.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from medimg_uq.models import build_model
from medimg_uq.train.config import ExperimentConfig, set_seed
from medimg_uq.train.losses import make_loss

LogFn = Callable[[dict], None]
MakeLoaders = Callable[[int], tuple[DataLoader, DataLoader]]


@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float


@dataclass
class TrainHistory:
    epochs: list[EpochStats]
    best_val_loss: float
    best_epoch: int
    checkpoint_path: str


def build_model_from_config(cfg: ExperimentConfig) -> nn.Module:
    """Construct the model described by ``cfg``."""
    return build_model(
        task=cfg.task,
        backbone=cfg.model.backbone,
        num_classes=cfg.num_classes,
        pretrained=cfg.model.pretrained,
        drop_rate=cfg.model.drop_rate,
        in_chans=cfg.model.in_chans,
    )


def _run_epoch_train(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.train()
    total_loss, total_n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(batch.images)
            loss = criterion(logits, batch.targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        n = len(batch)
        total_loss += loss.item() * n
        total_n += n
    return total_loss / max(total_n, 1)


@torch.no_grad()
def _run_epoch_val(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.images)
        loss = criterion(logits, batch.targets)
        n = len(batch)
        total_loss += loss.item() * n
        total_correct += (logits.argmax(dim=1) == batch.targets).sum().item()
        total_n += batch.targets.numel()
    n_examples = max(len(loader.dataset), 1)  # type: ignore[arg-type]
    return total_loss / n_examples, total_correct / max(total_n, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    cfg: ExperimentConfig,
    device: torch.device | str = "cpu",
    log_fn: LogFn | None = None,
) -> TrainHistory:
    """Train one model, keeping the best checkpoint by validation loss.

    The returned model has the best checkpoint's weights loaded back in, so the
    object you pass in is the best one you get out.
    """
    device = torch.device(device)
    model.to(device)
    use_amp = cfg.optim.amp and device.type == "cuda"

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )
    scheduler = None
    if cfg.optim.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.optim.epochs)
    criterion = make_loss(cfg.task, num_classes=cfg.num_classes).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_dir = Path(cfg.ckpt_dir) / cfg.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best.pt"

    history: list[EpochStats] = []
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(cfg.optim.epochs):
        train_loss = _run_epoch_train(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        val_loss, val_acc = _run_epoch_val(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        stats = EpochStats(
            epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_accuracy=val_acc
        )
        history.append(stats)
        if log_fn is not None:
            log_fn(vars(stats))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "config_name": cfg.name,
                    "backbone": cfg.model.backbone,
                    "num_classes": cfg.num_classes,
                },
                ckpt_path,
            )

    if best_epoch < 0:  # no epochs ran; save the untrained model so a checkpoint exists
        torch.save({"model_state": model.state_dict(), "epoch": -1}, ckpt_path)
    else:
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])

    return TrainHistory(
        epochs=history,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        checkpoint_path=str(ckpt_path),
    )


def train_ensemble(
    make_loaders: MakeLoaders,
    *,
    cfg: ExperimentConfig,
    device: torch.device | str = "cpu",
    log_fn: LogFn | None = None,
) -> list[str]:
    """Train ``cfg.ensemble_size`` members, each from a different seed.

    ``make_loaders(seed)`` returns the train and validation loaders for a member;
    tying it to the seed lets each member see a differently shuffled stream, which
    along with the different weight init is the source of ensemble diversity.
    Returns the checkpoint path of each member.
    """
    paths: list[str] = []
    for i in range(cfg.ensemble_size):
        seed = cfg.seed + i
        set_seed(seed)
        train_loader, val_loader = make_loaders(seed)
        member_cfg = replace(cfg, name=f"{cfg.name}/member_{i}", seed=seed)
        model = build_model_from_config(member_cfg)
        history = train_model(
            model, train_loader, val_loader, cfg=member_cfg, device=device, log_fn=log_fn
        )
        paths.append(history.checkpoint_path)
    return paths
