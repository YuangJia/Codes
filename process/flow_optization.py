import argparse
import logging
import os
import time
from collections import defaultdict, namedtuple
from itertools import accumulate
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import FastGeodis
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== 颜色映射（跟你原来一样） =====================

DEFAULT_TRANSITIONS = (15, 6, 4, 11, 13, 6)

def make_colorwheel(transitions: tuple = DEFAULT_TRANSITIONS) -> np.ndarray:
    colorwheel_length = sum(transitions)
    base_hues = map(
        np.array,
        (
            [255, 0, 0],
            [255, 255, 0],
            [0, 255, 0],
            [0, 255, 255],
            [0, 0, 255],
            [255, 0, 255],
            [255, 0, 0],
        ),
    )
    colorwheel = np.zeros((colorwheel_length, 3), dtype="uint8")
    hue_from = next(base_hues)
    start_index = 0
    for hue_to, end_index in zip(base_hues, accumulate(transitions)):
        transition_length = end_index - start_index
        colorwheel[start_index:end_index] = np.linspace(
            hue_from, hue_to, transition_length, endpoint=False
        )
        hue_from = hue_to
        start_index = end_index
    return colorwheel


def flow_to_rgb(
    flow: np.ndarray, flow_max_radius: Optional[float] = None, background: str = "bright"
) -> np.ndarray:
    valid_backgrounds = ("bright", "dark")
    if background not in valid_backgrounds:
        raise ValueError(f"background should be one of {valid_backgrounds}, got {background}.")
    wheel = make_colorwheel()
    complex_flow = flow[..., 0] + 1j * flow[..., 1]
    radius, angle = np.abs(complex_flow), np.angle(complex_flow)
    if flow_max_radius is None:
        flow_max_radius = np.max(radius)
    if flow_max_radius > 0:
        radius /= flow_max_radius
    ncols = len(wheel)
    angle[angle < 0] += 2 * np.pi
    angle = angle * ((ncols - 1) / (2 * np.pi))

    wheel = np.vstack((wheel, wheel[0]))
    (angle_fractional, angle_floor), angle_ceil = np.modf(angle), np.ceil(angle)
    angle_fractional = angle_fractional.reshape(angle_fractional.shape + (1,))
    float_hue = (
        wheel[angle_floor.astype(np.int32)] * (1 - angle_fractional)
        + wheel[angle_ceil.astype(np.int32)] * angle_fractional
    )

    ColorizationArgs = namedtuple(
        "ColorizationArgs", ["move_hue_valid_radius", "move_hue_oversized_radius", "invalid_color"]
    )

    def move_hue_on_V_axis(hues, factors):
        return hues * np.expand_dims(factors, -1)

    def move_hue_on_S_axis(hues, factors):
        return 255.0 - np.expand_dims(factors, -1) * (255.0 - hues)

    if background == "dark":
        parameters = ColorizationArgs(
            move_hue_on_V_axis, move_hue_on_S_axis, np.array([255, 255, 255], dtype=np.float32)
        )
    else:
        parameters = ColorizationArgs(
            move_hue_on_S_axis, move_hue_on_V_axis, np.array([0, 0, 0], dtype=np.float32)
        )

    colors = parameters.move_hue_valid_radius(float_hue, radius)
    oversized_radius_mask = radius > 1
    colors[oversized_radius_mask] = parameters.move_hue_oversized_radius(
        float_hue[oversized_radius_mask],
        1 / radius[oversized_radius_mask],
    )
    return colors.astype(np.uint8)


# ===================== 简单可视化（可选） =====================

def custom_draw_geometry_with_key_callback(pcds):
    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([76 / 255, 86 / 255, 106 / 255])
        return False

    key_to_callback = {ord("K"): change_background_to_black}
    o3d.visualization.draw_geometries_with_key_callbacks(pcds, key_to_callback)


# ===================== 计时器 =====================

class Timers(object):
    def __init__(self):
        self.timers = defaultdict(Timer)

    def tic(self, key):
        self.timers[key].tic()

    def toc(self, key):
        self.timers[key].toc()

    def print(self, key=None):
        if key is None:
            for k, v in self.timers.items():
                print(f"Average time for {k}: {v.avg():.6f}s")
        else:
            print(f"Average time for {key}: {self.timers[key].avg():.6f}s")

    def get_avg(self, key):
        return self.timers[key].avg()


class Timer(object):
    def __init__(self):
        self.reset()

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1

    def total(self):
        return self.total_time

    def avg(self):
        return self.total_time / float(self.calls) if self.calls > 0 else 0.0

    def reset(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0


# ===================== EarlyStopping（可选，保留） =====================

class EarlyStopping(object):
    def __init__(self, mode="min", min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True
        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)


# ===================== Distance Transform 类 =====================

class DT:
    def __init__(self, pts, pmin, pmax, grid_factor, device="cuda:0"):
        self.device = device
        self.grid_factor = grid_factor

        sample_x = ((pmax[0] - pmin[0]) * grid_factor).ceil().int() + 2
        sample_y = ((pmax[1] - pmin[1]) * grid_factor).ceil().int() + 2
        sample_z = ((pmax[2] - pmin[2]) * grid_factor).ceil().int() + 2

        self.Vx = (
            torch.linspace(0, sample_x, sample_x + 1, device=self.device)[:-1] /
            grid_factor + pmin[0]
        )
        self.Vy = (
            torch.linspace(0, sample_y, sample_y + 1, device=self.device)[:-1] /
            grid_factor + pmin[1]
        )
        self.Vz = (
            torch.linspace(0, sample_z, sample_z + 1, device=self.device)[:-1] /
            grid_factor + pmin[2]
        )

        grid_x, grid_y, grid_z = torch.meshgrid(self.Vx, self.Vy, self.Vz, indexing="ij")
        self.grid = torch.stack(
            [grid_x.unsqueeze(-1), grid_y.unsqueeze(-1), grid_z.unsqueeze(-1)], -1
        ).float().squeeze()
        H, W, D, _ = self.grid.size()
        pts_mask = torch.ones(H, W, D, device=device)

        self.pts_sample_idx_x = ((pts[:, 0:1] - self.Vx[0]) * self.grid_factor).round()
        self.pts_sample_idx_y = ((pts[:, 1:2] - self.Vy[0]) * self.grid_factor).round()
        self.pts_sample_idx_z = ((pts[:, 2:3] - self.Vz[0]) * self.grid_factor).round()
        pts_mask[
            self.pts_sample_idx_x.long(),
            self.pts_sample_idx_y.long(),
            self.pts_sample_idx_z.long(),
        ] = 0.0

        iterations = 1
        image_pts = torch.zeros(H, W, D, device=device).unsqueeze(0).unsqueeze(0)
        pts_mask = pts_mask.unsqueeze(0).unsqueeze(0)

        self.D = FastGeodis.generalised_geodesic3d(
            image_pts,
            pts_mask,
            [1.0 / self.grid_factor, 1.0 / self.grid_factor, 1.0 / self.grid_factor],
            1e10,
            0.0,
            iterations,
        ).squeeze()

    def torch_bilinear_distance(self, Y):
        H, W, D = self.D.size()
        target = self.D[None, None, ...]

        sample_x = ((Y[:, 0:1] - self.Vx[0]) * self.grid_factor).clip(0, H - 1)
        sample_y = ((Y[:, 1:2] - self.Vy[0]) * self.grid_factor).clip(0, W - 1)
        sample_z = ((Y[:, 2:3] - self.Vz[0]) * self.grid_factor).clip(0, D - 1)

        sample = torch.cat([sample_x, sample_y, sample_z], -1)
        sample = 2 * sample
        sample[..., 0] = sample[..., 0] / (H - 1)
        sample[..., 1] = sample[..., 1] / (W - 1)
        sample[..., 2] = sample[..., 2] / (D - 1)
        sample = sample - 1

        # grid_sample 要求顺序是 (z, y, x)
        sample_ = torch.cat(
            [sample[..., 2:3], sample[..., 1:2], sample[..., 0:1]], -1
        )

        dist = F.grid_sample(
            target,
            sample_.view(1, -1, 1, 1, 3),
            mode="bilinear",
            align_corners=True,
        ).view(-1)
        return dist


# ===================== MLP 神经先验 =====================

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class Neural_Prior(nn.Module):
    def __init__(
        self,
        dim_x=3,
        filter_size=128,
        act_fn="relu",
        layer_size=8,
    ):
        super().__init__()
        self.layer_size = layer_size

        self.nn_layers = nn.ModuleList([])
        if layer_size >= 1:
            # input layer
            self.nn_layers.append(nn.Linear(dim_x, filter_size))
            if act_fn == "relu":
                self.nn_layers.append(nn.ReLU())
            elif act_fn == "sigmoid":
                self.nn_layers.append(nn.Sigmoid())

            # hidden layers
            for _ in range(layer_size - 1):
                self.nn_layers.append(nn.Linear(filter_size, filter_size))
                if act_fn == "relu":
                    self.nn_layers.append(nn.ReLU())
                elif act_fn == "sigmoid":
                    self.nn_layers.append(nn.Sigmoid())

            # output: 3D flow
            self.nn_layers.append(nn.Linear(filter_size, dim_x))
        else:
            self.nn_layers.append(nn.Linear(dim_x, dim_x))

    def forward(self, x):
        # x: [1, N, 3]
        B, N, C = x.shape
        out = x
        for layer in self.nn_layers:
            # 打平做线性层，再 reshape 回去
            if isinstance(layer, nn.Linear):
                out = layer(out.view(-1, out.shape[-1])).view(B, N, -1)
            else:
                out = layer(out)
        return out


# ===================== 无 GT 的 solver =====================

def solver_no_gt(
    pc1: torch.Tensor,
    pc2: torch.Tensor,
    options: argparse.Namespace,
    net: nn.Module,
    max_iters: int,
):
    """
    pc1, pc2: [1, N, 3]
    目标：优化 net，使 pc1 + flow_pred 贴近 pc2（通过 DT loss）
    """
    if options.time:
        timers = Timers()
        timers.tic("solver_timer")

    pre_compute_st = time.time()

    if options.init_weight:
        net.apply(init_weights)
    for p in net.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(net.parameters(), lr=options.lr, weight_decay=0)

    total_losses = []

    if options.earlystopping:
        early_stopping = EarlyStopping(
            patience=options.early_patience,
            min_delta=options.early_min_delta,
            mode="min",
        )

    # ===== 构建 DT 区域 =====
    dt_start_time = time.time()

    pc1_min = torch.min(pc1.squeeze(0), 0)[0]
    pc2_min = torch.min(pc2.squeeze(0), 0)[0]
    pc1_max = torch.max(pc1.squeeze(0), 0)[0]
    pc2_max = torch.max(pc2.squeeze(0), 0)[0]

    xmin_int, ymin_int, zmin_int = torch.floor(
        torch.where(pc1_min < pc2_min, pc1_min, pc2_min) * options.grid_factor - 1
    ) / options.grid_factor
    xmax_int, ymax_int, zmax_int = torch.ceil(
        torch.where(pc1_max > pc2_max, pc1_max, pc2_max) * options.grid_factor + 1
    ) / options.grid_factor

    logging.info(
        f"xmin: {xmin_int}, xmax: {xmax_int}, "
        f"ymin: {ymin_int}, ymax: {ymax_int}, "
        f"zmin: {zmin_int}, zmax: {zmax_int}"
    )

    dt = DT(
        pc2.clone().squeeze(0).to(options.device),
        (xmin_int, ymin_int, zmin_int),
        (xmax_int, ymax_int, zmax_int),
        options.grid_factor,
        options.device,
    )

    dt_time = time.time() - dt_start_time

    pc1 = pc1.to(options.device).contiguous()
    pc2 = pc2.to(options.device).contiguous()

    pre_compute_time = time.time() - pre_compute_st

    best_loss = 1e10
    best_flow = None

    for epoch in range(max_iters):
        optimizer.zero_grad()

        flow_pred = net(pc1)             # [1, N, 3]
        pc1_deformed = pc1 + flow_pred   # [1, N, 3]

        loss = dt.torch_bilinear_distance(pc1_deformed.squeeze(0)).mean()
        loss.backward()
        optimizer.step()

        total_losses.append(loss.item())

        if loss.item() <= best_loss:
            best_loss = loss.item()
            best_flow = (pc1_deformed - pc1).detach().clone()

        if options.earlystopping:
            if early_stopping.step(loss):
                logging.info(f"Early stopping at iter {epoch}")
                break

        if epoch % 50 == 0:
            logging.info(f"[Iter {epoch}] Loss: {loss.item():.6f}")

    if options.time:
        timers.toc("solver_timer")
        time_avg = timers.get_avg("solver_timer")
        timers.print()
    else:
        time_avg = -1

    info_dict = {
        "final_flow": best_flow,          # [1, N, 3]
        "loss": best_loss,
        "build_dt_time": dt_time,
        "pre_compute_time": pre_compute_time,
        "time": time_avg,
    }

    if options.visualize:
        fig = plt.figure(figsize=(10, 4))
        ax = fig.gca()
        ax.plot(total_losses, label="loss")
        ax.legend()
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Loss vs iterations")
        plt.show()

        # 可视化预测 flow 上色后的点云
        pc1_o3d_pred = o3d.geometry.PointCloud()
        pc1_np = pc1[0].detach().cpu().numpy()
        flow_np = info_dict["final_flow"][0].detach().cpu().numpy()
        colors_flow = flow_to_rgb(flow_np)
        pc1_o3d_pred.points = o3d.utility.Vector3dVector(pc1_np)
        pc1_o3d_pred.colors = o3d.utility.Vector3dVector(colors_flow / 255.0)
        custom_draw_geometry_with_key_callback([pc1_o3d_pred])

    return info_dict


# ===================== 主函数：两个 ply 输入 =====================

def main_optimization(pc1_np,pc2_np):
    parser = argparse.ArgumentParser(description="Neural Scene Flow without GT (two PLYs).")

    parser.add_argument("--output_dir", type=str, default="./output_nogt", help="Where to save results.")
    parser.add_argument("--num_points", type=int, default=2048, help="Number of points to sample per cloud.")
    parser.add_argument("--iters", type=int, default=5000, help="Optimization iterations.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda or cpu.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--time", action="store_true", default=True)
    parser.add_argument("--use_all_points", action="store_true", default=True)
    parser.add_argument("--early_patience", type=int, default=500)
    parser.add_argument("--early_min_delta", type=float, default=1e-4)
    parser.add_argument("--init_weight", action="store_true", default=True)
    parser.add_argument("--earlystopping", action="store_true", default=True)

    # Neural prior
    parser.add_argument("--hidden_units", type=int, default=128)
    parser.add_argument("--layer_size", type=int, default=8)
    parser.add_argument("--act_fn", type=str, default="relu")

    # DT grid
    parser.add_argument("--grid_factor", type=float, default=10.0)

    # options = parser.parse_args()
    options = parser.parse_args([])

    os.makedirs(options.output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(options.output_dir, "run.log")),
            logging.StreamHandler(),
        ],
    )
    logging.info(options)

    torch.manual_seed(options.seed)
    np.random.seed(options.seed)
    if "cuda" in options.device:
        torch.cuda.manual_seed_all(options.seed)

    # ===== 读取 ply 点云 =====
    # pcd1 = o3d.io.read_point_cloud(options.pc1_path)
    # pcd2 = o3d.io.read_point_cloud(options.pc2_path)

    # pc1_np = np.asarray(pcd1.points, dtype=np.float32)
    # pc2_np = np.asarray(pcd2.points, dtype=np.float32)

    logging.info(f"pc1: {pc1_np.shape}, pc2: {pc2_np.shape}")

    # 对齐采样数量
    if not options.use_all_points:
        N = min(pc1_np.shape[0], pc2_np.shape[0], options.num_points)
        idx1 = np.random.choice(pc1_np.shape[0], N, replace=False)
        idx2 = np.random.choice(pc2_np.shape[0], N, replace=False)
        pc1_np = pc1_np[idx1]
        pc2_np = pc2_np[idx2]
        logging.info(f"Using {N} points per cloud.")
    else:
        N = min(pc1_np.shape[0], pc2_np.shape[0])
        pc1_np = pc1_np[:N]
        pc2_np = pc2_np[:N]
        logging.info(f"Using ALL points, truncated to {N}.")

    # pc1 = torch.from_numpy(pc1_np).unsqueeze(0)  # [1, N, 3]
    # pc2 = torch.from_numpy(pc2_np).unsqueeze(0)  # [1, N, 3]
    pc1 = torch.from_numpy(pc1_np).unsqueeze(0).float().to(options.device)
    pc2 = torch.from_numpy(pc2_np).unsqueeze(0).float().to(options.device)

    # ===== 初始化网络 =====
    net = Neural_Prior(
        dim_x=3,
        filter_size=options.hidden_units,
        act_fn=options.act_fn,
        layer_size=options.layer_size,
    ).to(options.device)

    # ===== 优化 =====
    info = solver_no_gt(pc1, pc2, options, net, options.iters)
    flow_pred = info["final_flow"][0].cpu().numpy()  # [N, 3]
    return flow_pred

    # ===== 保存结果 =====
    # np.save(os.path.join(options.output_dir, "flow_pred.npy"), flow_pred)
    # logging.info(f"Saved flow_pred.npy to {options.output_dir}")

    # 保存上色后的点云
    # colors = flow_to_rgb(flow_pred)
    # pcd_out = o3d.geometry.PointCloud()
    # pcd_out.points = o3d.utility.Vector3dVector(pc1_np)
    # pcd_out.colors = o3d.utility.Vector3dVector(colors / 255.0)
    # o3d.io.write_point_cloud(
    #     os.path.join(options.output_dir, "pc1_with_pred_flow.ply"),
    #     pcd_out,
    # )
    # logging.info("Saved pc1_with_pred_flow.ply with flow colors.")


"""
python flow_optization.py \
    --pc1_path /data2/ljl/jya_repo/4D/Pi3/nuscenes/103/key_frame/frame_0000.ply \
    --pc2_path /data2/ljl/jya_repo/4D/Pi3/nuscenes/103/key_frame/frame_0001.ply \
    --output_dir /data2/ljl/jya_repo/4D/Pi3/nuscenes/103/flow_pred \
    --use_all_points \
    --iters 5000 \
    --device cuda:0
    
python flow_optization.py \
    --pc1_path /data2/ljl/jya_repo/4D/Pi3/nuscenes/103/key_frame/frame_0000.ply \
    --pc2_path /data2/ljl/jya_repo/4D/Pi3/nuscenes/103/key_frame/frame_0001.ply \
    --output_dir /data2/ljl/jya_repo/4D/Pi3/nuscenes/103/flow_pred \
    --num_points 4096 \
    --iters 5000 \
    --device cuda:0
"""