from typing import Dict
import torch


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    valid_mask = torch.nan_to_num(valid_mask, nan=0.0, posinf=0.0, neginf=0.0)

    sq_error = (pred - target) ** 2
    sq_error = sq_error * valid_mask
    return sq_error.sum() / (valid_mask.sum() + eps)


def masked_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    mse = masked_mse(pred, target, valid_mask, eps=eps)
    return torch.sqrt(mse + eps)


def masked_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    valid_mask = torch.nan_to_num(valid_mask, nan=0.0, posinf=0.0, neginf=0.0)

    abs_error = torch.abs(pred - target)
    abs_error = abs_error * valid_mask
    return abs_error.sum() / (valid_mask.sum() + eps)


def masked_error_std(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    valid_mask = torch.nan_to_num(valid_mask, nan=0.0, posinf=0.0, neginf=0.0)

    residual = (pred - target) * valid_mask
    n = valid_mask.sum()

    mean_residual = residual.sum() / (n + eps)
    var = (((residual - mean_residual) * valid_mask) ** 2).sum() / (n + eps)
    return torch.sqrt(var + eps)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_fn=None,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_sq_error = 0.0
    total_abs_error = 0.0
    total_residual = 0.0
    total_residual_sq = 0.0
    total_valid = 0.0
    num_batches = 0

    nb = device.type == "cuda"
    for batch in dataloader:
        image = batch["image"].to(device, non_blocking=nb)
        depth = batch["depth"].to(device, non_blocking=nb)
        valid_mask = batch["valid_mask"].to(device, non_blocking=nb)

        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        valid_mask = torch.nan_to_num(valid_mask, nan=0.0, posinf=0.0, neginf=0.0)

        pred = model(image)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

        if loss_fn is not None:
            loss = loss_fn(pred, depth, valid_mask)
            total_loss += loss.item()

        residual = pred - depth
        sq_error = (residual ** 2) * valid_mask
        abs_error = residual.abs() * valid_mask

        total_sq_error += sq_error.sum().item()
        total_abs_error += abs_error.sum().item()
        total_residual += (residual * valid_mask).sum().item()
        total_residual_sq += ((residual ** 2) * valid_mask).sum().item()
        total_valid += valid_mask.sum().item()
        num_batches += 1

    if total_valid == 0:
        raise ValueError("No valid pixels found during evaluation.")

    mse = total_sq_error / total_valid
    rmse = mse ** 0.5
    mae = total_abs_error / total_valid

    mean_residual = total_residual / total_valid
    var_residual = max(total_residual_sq / total_valid - mean_residual ** 2, 0.0)
    std = var_residual ** 0.5

    results = {
        "rmse": rmse,
        "mae": mae,
        "std": std,
        "num_valid_pixels": total_valid,
    }

    if loss_fn is not None and num_batches > 0:
        results["loss"] = total_loss / num_batches

    return results


if __name__ == "__main__":
    pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target = torch.tensor([[[[1.5, 2.5], [2.5, 3.5]]]])
    valid_mask = torch.tensor([[[[1.0, 1.0], [1.0, 0.0]]]])

    print("MSE :", masked_mse(pred, target, valid_mask).item())
    print("RMSE:", masked_rmse(pred, target, valid_mask).item())
    print("MAE :", masked_mae(pred, target, valid_mask).item())
    print("STD :", masked_error_std(pred, target, valid_mask).item())