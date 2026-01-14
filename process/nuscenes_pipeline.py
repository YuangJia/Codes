import torch
import argparse
from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3
import os
import glob
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
import copy
import gc
import tempfile, shutil
import open3d as o3d
import sys
import cv2
from tqdm import tqdm
import yaml
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import json
import importlib
import numpy as np
from typing import List
from flow_optization import main_optimization
import glob
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GSAM_PATH = os.path.join(ROOT, "Grounded-SAM-2")
sys.path.append(GSAM_PATH)
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo



cam_keys = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT'
]

cam2id = {'CAM_FRONT_LEFT': 0, 'CAM_FRONT': 1, 'CAM_FRONT_RIGHT': 2, 'CAM_BACK_RIGHT': 3, 'CAM_BACK': 4, 'CAM_BACK_LEFT': 5}


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)   # 包含 numpy.ndarray 和 list


def load_image_same_resize(img_path, PIXEL_LIMIT=255000, return_numpy=False):
    img = Image.open(img_path).convert("RGB")

    W_orig, H_orig = img.size
    scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target, H_target = W_orig * scale, H_orig * scale

    k, m = round(W_target / 14), round(H_target / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > W_target / H_target:
            k -= 1
        else:
            m -= 1

    TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14

    resized = img.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)

    if return_numpy:
        return np.array(resized)
    return resized

def apply_mask(pts, cols,  pix, cam, mask):
    return (
        pts[mask],
        cols[mask],
        pix[mask],
        cam[mask]
    )

class NuScenesDataset:
    def __init__(
        self,
        nusc_root_dir: str,
        sequence: int,
        *_,
        **__,
    ):
        # Import nuscenes
        try:
            importlib.import_module("nuscenes")
        except ModuleNotFoundError:
            print("nuscenes-devkit is not installed.")
            print('Run "pip install nuscenes-devkit"')
            sys.exit(1)

        from nuscenes.nuscenes import NuScenes

        # Use mini or trainval? → you used mini, keep it
        nusc_version: str = "v1.0-mini"

        # Load NuScenes
        self.nusc = NuScenes(dataroot=str(nusc_root_dir), version=nusc_version)

        # Build scene name
        self.sequence_id = str(int(sequence)).zfill(4)
        self.scene_name = f"scene-{self.sequence_id}"

        # Check scene exists
        all_scene_names = [s["name"] for s in self.nusc.scene]
        if self.scene_name not in all_scene_names:
            print(f'[ERROR] Scene "{self.scene_name}" not found in {nusc_version}.')
            print("Available scenes:")
            self.nusc.list_scenes()
            sys.exit(1)

        # Retrieve scene token
        self.scene_token = [
            s["token"] for s in self.nusc.scene if s["name"] == self.scene_name
        ][0]

        # Load all sample_tokens in this scene (2Hz keyframes)
        self.sample_tokens = self._get_sample_tokens(self.scene_token)

    def __len__(self):
        return len(self.sample_tokens)

    # ---------------------------------------------------------------
    # Load all sample tokens (keyframes) for this scene in order
    # ---------------------------------------------------------------
    def _get_sample_tokens(self, scene_token: str) -> List[str]:
        scene_rec = self.nusc.get("scene", scene_token)
        sample_token = scene_rec["first_sample_token"]

        sample_tokens = []
        while sample_token != "":
            sample_tokens.append(sample_token)
            sample_rec = self.nusc.get("sample", sample_token)
            sample_token = sample_rec["next"]

        return sample_tokens

    # ---------------------------------------------------------------
    # For each sample, return 6 camera images + K + world extrinsic
    # ---------------------------------------------------------------
    def get_camera_info(self, sample_token):
        cam_infos = {}
        sample = self.nusc.get("sample", sample_token)

        for cam in cam_keys:
            cam_token = sample['data'][cam]
            cam_sd = self.nusc.get("sample_data", cam_token)

            cs_record = self.nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
            ep_record = self.nusc.get("ego_pose", cam_sd["ego_pose_token"])

            # World extrinsic = ego_pose × calibrated_sensor
            extrinsic_world = (
                self._pose_matrix(ep_record["translation"], ep_record["rotation"])
                @ self._pose_matrix(cs_record["translation"], cs_record["rotation"])
            )

            intrinsics = np.array(cs_record["camera_intrinsic"], dtype=np.float32)
            image_path = os.path.join(self.nusc.dataroot, cam_sd["filename"])

            cam_infos[cam] = {
                "image": image_path,
                "K": intrinsics,
                "extrinsic": extrinsic_world
            }

        return cam_infos

    # ---------------------------------------------------------------
    def _pose_matrix(self, translation, rotation):
        from nuscenes.utils.geometry_utils import transform_matrix
        return transform_matrix(translation, Quaternion(rotation))


# =================== 主程序 ===================

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Pi3 + Flow + Cluster Filtering (all in memory)")
    
    # Pi3 相关参数
    parser.add_argument("--save_path", type=str, default='examples/skating',
                        help="Path to save some outputs (e.g., camera poses, final pcd).")
    parser.add_argument("--interval", type=int, default=1,
                        help="Interval to sample image. Default: 1 for images dir, 10 for video")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to the Pi3 model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
    
    # Flow 配置文件
    parser.add_argument("--flow_cfg", type=str, default="/data2/ljl/jya_repo/4D/Pi3/let_it_flow/new_config.yaml",
                        help="Config file for flow (lif / chodosh / NP / MBNSFP etc.)")
    
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    # =================== 读取 Flow 配置 ===================
    with open(args.flow_cfg) as file:
        config_file = yaml.load(file, Loader=yaml.FullLoader)
    cfg = config_file['cfg']

    use_gpu = 0
    if torch.cuda.is_available():
        flow_device = torch.device(cfg['device'])
    else:
        flow_device = torch.device('cpu')

    # model_name = cfg['model']          # 'lif' / 'chodosh' / ...
    # frame = cfg['frame']              # 起始帧 id，一般为 0
    # dist_w = cfg['dist_w']
    # TEMPORAL_RANGE = cfg['TEMPORAL_RANGE']   # 例如 2
    # sc_w = cfg['sc_w']
    # trunc_dist = cfg['trunc_dist']
    # passing_ids = cfg['passing_ids']
    # K_ = cfg['K']
    # d_thre = cfg['d_thre']
    # eps = cfg['eps']
    # min_samples = cfg['min_samples']
    # lr = cfg['lr']
    # d_min_height = cfg['min_height']

    # =================== 1. 准备 NuScenes 图像和相机参数 ===================
    print("Loading NuScenes meta ...")
    nuscenes = NuScenesDataset(
        nusc_root_dir='/data2/DATA/AutoDrive/nuscenes/v1.0-mini',
        sequence=103
    )
    
    images_path_list = [[] for _ in range(6)]
    intrinsics = [[] for _ in range(6)]
    extrinsics = [[] for _ in range(6)]
    
    for t, sample_token in enumerate(nuscenes.sample_tokens):
        cam_info = nuscenes.get_camera_info(sample_token)
        for cam, info in cam_info.items():
            images_path_list[cam2id[cam]].append(info['image'])
            K = info["K"]
            intrinsics[cam2id[cam]].append(K)
            T_cam_world = info["extrinsic"]
            extrinsics[cam2id[cam]].append(T_cam_world)
    
    # 你原来是 range(18,21)，这里要保证：N_cam * TEMPORAL_RANGE == 选出来的总帧数
    # 例如 TEMPORAL_RANGE=2，N_cam=6，就要 12 张图，这里示例用 frame_id 18 和 19 两帧：
    N_cam = 6
    T = 2   # 和 flow 配置保持一致
    chosen_frames = [18, 19]    # 示例：两帧
    assert len(chosen_frames) == T, "chosen_frames 的数量要等于 TEMPORAL_RANGE"

    processed_imgs = []
    processed_k = []
    processed_poses = []
    for frame_id in chosen_frames:
        for cam in range(N_cam):
            processed_imgs.append(images_path_list[cam][frame_id])
            processed_k.append(intrinsics[cam][frame_id])
            processed_poses.append(extrinsics[cam][frame_id])

    # =================== 2. Pi3 模型加载与推理 ===================
    print(f"Loading Pi3 model...")
    device = torch.device(args.device)
    if args.ckpt is not None:
        model = Pi3().to(device).eval()
        if args.ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(args.ckpt)
        else:
            weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(weight)
    else:
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()

    # 载入图像
    imgs = load_images_as_tensor(processed_imgs, interval=args.interval).to(device)  # (N, 3, H, W)
    N_img, C_, H_, W_ = imgs.shape 
    print(f"imgs.shape:{imgs.shape}")

    # --- 1. 创建临时目录 ---
    tmp_dir = tempfile.mkdtemp(prefix="pi3_sam2_frames_")
    print("Temporary directory for SAM2:", tmp_dir)

    # --- 2. 保存imgs为图像 ---
    import torchvision.transforms.functional as TF

    for idx in range(imgs.shape[0]):
        # imgs 是 tensor [N,3,H,W] 且范围为[0,1]
        if idx not in [1,7]:
            continue
        img = TF.to_pil_image(imgs[idx].cpu())
        save_path = os.path.join(tmp_dir, f"{idx:06d}.jpg")
        img.save(save_path)
    
    print("Running Pi3 inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            res = model(imgs[None])  # (1, N, H, W, ...)

    scale = 30.0
    del model
    torch.cuda.empty_cache()
    # 一些中间量
    points_all = res['points'][0]                  # (N, H, W, 3)
    depth_all  = res['local_points'][0][..., 2]    # (N, H, W)
    colors_all = imgs.permute(0, 2, 3, 1)          # (N, H, W, 3)

    # R_align （和你原来的逻辑一样）
    first_left_cam = res['camera_poses'][0][0][:3, :3].cpu().numpy()
    first_left_cam_gt = extrinsics[0][0][:3, :3]
    R_align = first_left_cam_gt @ first_left_cam.T

    # =================== 3. 在内存中构造 global_list / global_color_list ===================
    global_list = []        # 每个元素: (Ni, 4) → (x, y, z, frame_idx)
    global_color_list = []  # 每个元素: (Ni, 4) → (r, g, b, frame_idx)

    sample_ratio = 0.6
    abs_distance = 60.0
    depth_keep_ratio = 0.45

    frame_pts_list = []
    frame_cols_list = []
    frame_pix_list = []
    frame_cam_list = []
    
    for t_idx in range(T):   # t_idx = 0,1
        start = t_idx * N_cam
        end   = (t_idx + 1) * N_cam

        frame_points_list = []
        frame_colors_list = []
        frame_pix_idx_list = []     # (K_i,)   torch，0..H*W-1
        frame_cam_idx_list = []      # (K_i,)   torch，0..N_total-1

        for i in range(start, end):
            pts  = points_all[i] * scale      # (H, W, 3)
            cols = colors_all[i]              # (H, W, 3)
            depth = depth_all[i]              # (H, W)

            H, W = pts.shape[:2]

            # 只保留图像下方 60%
            cut_row = int(H * 0.4)
            keep_mask_2d = np.zeros((H, W), dtype=bool)
            keep_mask_2d[cut_row:, :] = True
            keep_mask = torch.from_numpy(keep_mask_2d.reshape(-1)).to(pts.device)

            pts_flat  = pts.reshape(-1, 3)
            cols_flat = cols.reshape(-1, 3)
            depth_flat = depth.reshape(-1)
            pix_idx_flat = torch.arange(H * W, device=pts.device)   # (HW,)
            cam_idx_flat = torch.full_like(pix_idx_flat, i)         # (HW,)

            pts_flat  = pts_flat[keep_mask]
            cols_flat = cols_flat[keep_mask]
            depth_flat = depth_flat[keep_mask]
            pix_idx_flat = pix_idx_flat[keep_mask]
            cam_idx_flat = cam_idx_flat[keep_mask]
            

            # 某几个相机不做 depth percentile 筛选
            if i not in [1, 4, 7, 10]:
                depth_np = depth_flat.cpu().numpy()
                th = np.percentile(depth_np, depth_keep_ratio * 100.0)
                depth_mask = depth_np <= th
                pts_flat  = pts_flat[depth_mask]
                cols_flat = cols_flat[depth_mask]
                pix_idx_flat = pix_idx_flat[depth_mask]
                cam_idx_flat = cam_idx_flat[depth_mask]

            frame_points_list.append(pts_flat)
            frame_colors_list.append(cols_flat)
            frame_pix_idx_list.append(pix_idx_flat)
            frame_cam_idx_list.append(cam_idx_flat)
            cam_id = i - start     # 把 6-camera block 映射成 [0..5]
            # frame_cam_idx_list.append(cam_idx_flat)

        # 合并 6 个 camera 的点云
        frame_points = torch.cat(frame_points_list, dim=0)   # (M, 3)
        frame_colors = torch.cat(frame_colors_list, dim=0)   # (M, 3)
        frame_pix_idx  = torch.cat(frame_pix_idx_list, dim=0)  # (M,)
        frame_cam_idx  = torch.cat(frame_cam_idx_list, dim=0)  # (M,)
        

        pts_np  = frame_points.cpu().numpy()
        cols_np = frame_colors.cpu().numpy()
        pix_np    = frame_pix_idx.cpu().numpy()     # (M,)
        cam_np    = frame_cam_idx.cpu().numpy()     # (M,)

        # 全局距离裁剪
        dist = np.linalg.norm(pts_np, axis=1)
        dist_mask = dist <= abs_distance
        pts_np  = pts_np[dist_mask]
        cols_np = cols_np[dist_mask]
        pix_np    = pix_np[dist_mask]
        cam_np    = cam_np[dist_mask]

        # 随机下采样
        n = pts_np.shape[0]
        n_keep = max(1, int(n * sample_ratio))
        sel = np.random.choice(n, n_keep, replace=False)
        pts_np  = pts_np[sel]
        cols_np = cols_np[sel]
        pix_np    = pix_np[sel]
        cam_np    = cam_np[sel]

        # 对齐旋转
        pts_np = (R_align @ pts_np.T).T
        
        # ---- NEW: 保存每帧点云 ----
        frame_pts_list.append(pts_np)
        frame_cols_list.append(cols_np)
        frame_pix_list.append(pix_np)
        frame_cam_list.append(cam_np)
    
    # ---------- NEW: 限制 frame0 点数 <= frame1 ----------
    if T == 2:
        pts0   = frame_pts_list[0]
        cols0  = frame_cols_list[0]
        pix0   = frame_pix_list[0]
        cam0   = frame_cam_list[0]

        pts1   = frame_pts_list[1]

        N0 = pts0.shape[0]
        N1 = pts1.shape[0]

        if N0 > N1:
            sel = np.random.choice(N0, N1, replace=False)
            mask = np.zeros(N0, dtype=bool)
            mask[sel] = True

            pts0, cols0, pix0, cam0 = apply_mask(
                pts0, cols0, pix0, cam0, mask
            )

        # pack back
        frame_pts_list[0]  = pts0
        frame_cols_list[0] = cols0
        frame_pix_list[0] = pix0
        frame_cam_list[0] = cam0      
    
    # 6. 对相机外参 camera_poses 进行尺度变换 & 旋转对齐
    cam_poses = res['camera_poses'][0].cpu().numpy()  # (N_total,4,4)
    # 缩放平移部分
    for i in range(cam_poses.shape[0]):
        cam_poses[i][:3, 3] *= scale

    cam_poses_aligned = cam_poses.copy()
    for i in range(cam_poses.shape[0]):
        T = cam_poses[i]        # 4x4 cam->world_est
        R = T[:3, :3]
        t = T[:3, 3]
        # Apply global rotation
        R_new = R_align @ R
        t_new = R_align @ t
        cam_poses_aligned[i, :3, :3] = R_new
        cam_poses_aligned[i, :3, 3]  = t_new

    pose_txt_path = f"{args.save_path}/camera_poses_aligned.txt"
    with open(pose_txt_path, "w") as f:
        for i in range(cam_poses_aligned.shape[0]):
            flat = cam_poses_aligned[i].reshape(-1)
            line = " ".join([f"{x:.8f}" for x in flat])
            f.write(line + "\n")
    print(f"[OK] Saved aligned camera poses to: {pose_txt_path}")
    
    print(f"frame_pts_list[0].shape:{frame_pts_list[0].shape}, frame_pts_list[1].shape:{frame_pts_list[1].shape},frame_pix_list[0].shape:{frame_pix_list[0].shape}")
    flow_pred = main_optimization(frame_pts_list[0],frame_pts_list[1])
    # print(f"flow_pred.shape:{flow_pred.shape}")
    flow_mag = np.linalg.norm(flow_pred, axis=1)
    dynamic_region_flow = 2.0
    flow_masks_0 = flow_mag >= dynamic_region_flow
    flow_masks_0= flow_masks_0.astype(bool)  
    # # --- 统一 numpy 类型 ---
    # pix_idx_np = frame_pix_idx_list[0].cpu().numpy()     # shape (N,), int
    # flow_masks_0_np = flow_masks_0.astype(bool)          # shape (N,), bool

    # assert pix_idx_np.shape[0] == flow_masks_0_np.shape[0], \
    #     f"Mask size mismatch: pix={pix_idx_np.shape}, flow={flow_masks_0_np.shape}"
    
    
    pts0, cols0,  pix0, cam0 = frame_pts_list[0], frame_cols_list[0], frame_pix_list[0], frame_cam_list[0]
    pts0, cols0,  pix0, cam0 = apply_mask(
        pts0, cols0, pix0, cam0, flow_masks_0
    )

    frame_pts_list[0]  = pts0
    frame_cols_list[0] = cols0
    frame_pix_list[0] = pix0
    frame_cam_list[0] = cam0


    # --- 根据flow筛选像素index ---
    # cur_mask_0 = pix_idx_np[flow_masks_0_np]   # 得到满足条件的像素编号
    pts0  = frame_pts_list[0]      # (N0, 3)
    cols0 = frame_cols_list[0]     # (N0, 3)
    pix0  = frame_pix_list[0]      # (N0,)
    cam0  = frame_cam_list[0]      # (N0,)
    cam0_mask = (cam0 == 1)
    pix_cam0 = pix0[cam0_mask]   # shape (N_cam0,)
    cur_mask_0 = np.zeros((H_, W_), dtype=bool)
    vs = pix_cam0 // W_
    us = pix_cam0 % W_
    cur_mask_0[vs, us] = True    
    
    
    # use bfloat16 for the entire notebook
    # cur_mask_0 = np.ones((H_, W_), dtype=bool) # for test
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # init sam image predictor and video predictor model
    sam2_checkpoint = "/data2/ljl/jya_repo/4D/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)

    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(sam2_image_model)


    # init grounding dino model from huggingface
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)


    # setup the input image and text prompt for SAM 2 and Grounding DINO
    # VERY important: text queries need to be lowercased + end with a dot
    text = "car."
    
    # scan all the JPEG frame names in this directory
    video_dir = tmp_dir
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    
    # init video predictor state
    inference_state = video_predictor.init_state(
        video_path=video_dir,
        offload_video_to_cpu=True,
        async_loading_frames=True
    )
    step = 20  # the step to sample frames for Grounding DINO predictor

    sam2_masks = MaskDictionaryModel()
    PROMPT_TYPE_FOR_VIDEO = "mask"  # box, mask or point
    objects_count = 0

    # 用来记录：哪些 object_id 是“高光流实例”，以及它们在 t=0 / t=1 的 mask
    highflow_object_ids = set()
    highflow_masks_t0 = {}  # {obj_id: mask0 (H,W,bool)}
    highflow_masks_t1 = {}  # {obj_id: mask1 (H,W,bool)}

    print("Total frames:", len(frame_names))
    for start_frame_idx in range(0, len(frame_names), step):
        print("start_frame_idx", start_frame_idx)

        img_path = os.path.join(video_dir, frame_names[start_frame_idx])
        image = Image.open(img_path)
        image_base_name = frame_names[start_frame_idx].split(".")[0]
        mask_dict = MaskDictionaryModel(
            promote_type=PROMPT_TYPE_FOR_VIDEO,
            mask_name=f"mask_{image_base_name}.npy"
        )

        # run Grounding DINO on the image
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = grounding_model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.25,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]]
        )

        # prompt SAM image predictor to get the mask for the object
        image_predictor.set_image(np.array(image.convert("RGB")))

        # process the detection results
        input_boxes = results[0]["boxes"]
        OBJECTS = results[0]["labels"]

        if input_boxes.shape[0] != 0:
            # prompt SAM 2 image predictor to get the mask for the object
            masks, scores, logits = image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            # convert the mask shape to (n, H, W)
            if masks.ndim == 2:
                masks = masks[None]
                scores = scores[None]
                logits = logits[None]
            elif masks.ndim == 4:
                masks = masks.squeeze(1)

            # ====== 这里加：在起始帧（假设是第 0 帧），用 cur_mask_0 筛选实例 ======
            if start_frame_idx == 0:
                masks_np = to_numpy(masks)  # (N_obj, H, W)
                keep_idx = []
                for k in range(masks_np.shape[0]):
                    inst_mask = masks_np[k].astype(bool)
                    if np.any(inst_mask & cur_mask_0):
                        keep_idx.append(k)

                if len(keep_idx) == 0:
                    print("[WARN] No SAM instance overlaps with high-flow region on frame 0.")
                    # 这里你可以选择 continue 或者保留全部，先简单 continue
                    continue

                masks = masks[keep_idx]
                input_boxes = input_boxes[keep_idx]
                OBJECTS = [OBJECTS[k] for k in keep_idx]

            """
            Step 3: Register each object's positive points to video predictor
            """

            if mask_dict.promote_type == "mask":
                mask_dict.add_new_frame_annotation(
                    mask_list=torch.tensor(masks).to(device),
                    box_list=torch.tensor(input_boxes),
                    label_list=OBJECTS
                )
            else:
                raise NotImplementedError("SAM 2 video predictor only support mask prompts")

            """
            Step 4: Propagate the video predictor to get the segmentation results for each frame
            """
            objects_count = mask_dict.update_masks(
                tracking_annotation_dict=sam2_masks,
                iou_threshold=0.8,
                objects_count=objects_count
            )
            print("objects_count", objects_count)

            # ====== 在起始帧，再用 mask_dict.labels 和 cur_mask_0 精确确定高光流实例 ID ======
            if start_frame_idx == 0:
                for obj_id, obj_info in mask_dict.labels.items():
                    mask0 = obj_info.mask  # 可能是 [1,H,W] 或 [H,W]
                    if isinstance(mask0, torch.Tensor):
                        mask0_np = mask0.detach().cpu().numpy()
                    else:
                        mask0_np = np.array(mask0)
                    if mask0_np.ndim == 3:
                        mask0_np = mask0_np[0]
                    mask0_np = mask0_np.astype(bool)

                    if np.any(mask0_np & cur_mask_0):
                        highflow_object_ids.add(obj_id)
                        highflow_masks_t0[obj_id] = mask0_np

        else:
            print("No object detected in the frame, skip merge the frame merge {}".format(frame_names[start_frame_idx]))
            mask_dict = sam2_masks

        if len(mask_dict.labels) == 0:
            mask_dict.save_empty_mask_and_json(
                mask_data_dir,
                json_data_dir,
                image_name_list=frame_names[start_frame_idx:start_frame_idx+step]
            )
            print("No object detected in the frame, skip the frame {}".format(start_frame_idx))
            continue
        else:
            video_predictor.reset_state(inference_state)

            # 只把“高光流实例”的 ID 提交给 video predictor 做 tracking
            for object_id, object_info in mask_dict.labels.items():
                if len(highflow_object_ids) > 0 and object_id not in highflow_object_ids:
                    continue
                frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                    inference_state,
                    start_frame_idx,
                    object_id,
                    object_info.mask,
                )

            video_segments = {}  # output the following {step} frames tracking masks
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
                inference_state,
                max_frame_num_to_track=step,
                start_frame_idx=start_frame_idx
            ):
                frame_masks = MaskDictionaryModel()

                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = (out_mask_logits[i] > 0.0)  # (1,H,W)
                    object_info = ObjectInfo(
                        instance_id=out_obj_id,
                        mask=out_mask[0],
                        class_name=mask_dict.get_target_class_name(out_obj_id)
                    )
                    object_info.update_box()
                    frame_masks.labels[out_obj_id] = object_info
                    image_base_name = frame_names[out_frame_idx].split(".")[0]
                    frame_masks.mask_name = f"mask_{image_base_name}.npy"
                    frame_masks.mask_height = out_mask.shape[-2]
                    frame_masks.mask_width = out_mask.shape[-1]

                video_segments[out_frame_idx] = frame_masks
                sam2_masks = copy.deepcopy(frame_masks)

            print("video_segments:", len(video_segments))

        """
        Step 5: save the tracking masks and json files
        """
        # for frame_idx, frame_masks_info in video_segments.items():
        #     mask = frame_masks_info.labels
        #     mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
        #     for obj_id, obj_info in mask.items():
        #         mask_img[obj_info.mask == True] = obj_id

        #     mask_img = mask_img.numpy().astype(np.uint16)
        #     np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)

        #     json_data = frame_masks_info.to_dict()
        #     json_data_path = os.path.join(
        #         json_data_dir,
        #         frame_masks_info.mask_name.replace(".npy", ".json")
        #     )
        #     with open(json_data_path, "w") as f:
        #         json.dump(json_data, f)

        # ====== 在第一帧 (start_frame_idx == 0) 处理高光流实例 ======
        if start_frame_idx == 0:
            # (1) 创建 t0 合并 mask
            merged_mask_t0 = np.zeros_like(cur_mask_0, dtype=bool)
            for obj_id, mask0 in highflow_masks_t0.items():
                merged_mask_t0 |= mask0   # OR 聚合所有实例

            # (2) 现在获得下一帧对应 ID 的 mask
            next_frame_idx = start_frame_idx + 1
            merged_mask_t1 = None

            if next_frame_idx in video_segments:
                merged_mask_t1 = np.zeros_like(cur_mask_0, dtype=bool)
                for obj_id, obj_info in video_segments[next_frame_idx].labels.items():
                    if obj_id in highflow_object_ids:
                        mask1 = obj_info.mask
                        if isinstance(mask1, torch.Tensor):
                            mask1_np = mask1.detach().cpu().numpy()
                        else:
                            mask1_np = np.array(mask1)
                        if mask1_np.ndim == 3:
                            mask1_np = mask1_np[0]

                        mask1_np = mask1_np.astype(bool)
                        merged_mask_t1 |= mask1_np   # OR 聚合
            else:
                print("[WARN] No mask found for next frame!!")

            # ===== 保存结果（可根据需要替换） =====
            # np.save(os.path.join(args.save_path, "merged_mask_t0.npy"), merged_mask_t0)
            # if merged_mask_t1 is not None:
            #     np.save(os.path.join(args.save_path, "merged_mask_t1.npy"), merged_mask_t1)
            mask_save_dir = os.path.join(args.save_path, "merged_masks")
            os.makedirs(mask_save_dir, exist_ok=True)
            mask_u8 = (merged_mask_t0.astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(mask_save_dir, "merged_mask_t0.png"), mask_u8)
            if merged_mask_t1 is not None:
                mask_u8_t1 = (merged_mask_t1.astype(np.uint8) * 255)
                cv2.imwrite(os.path.join(mask_save_dir, "merged_mask_t1.png"), mask_u8_t1)
            print("[OK] Saved merged instance masks for frame 0 and frame 1.")

    print("[INFO] Cleaning up tmp_dir...")
    shutil.rmtree(tmp_dir)
