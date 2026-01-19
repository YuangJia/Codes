"""
conda create -n drive --clone PyTorch-2.1.0
conda activate drive
pip install einops  scipy==1.12.0 imageio pillow opencv-python==4.9.0.80  numpy==1.23.0


python get_batch_demo.py


demo脚本: 从 nuscenes 数据集取一个batch (16帧 x 6个摄像头+逐帧相对位姿)
"""
import os
import sys
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
import json


from dataset.nuscenes import TrainDataset16Frames, ValDataset16Frames



def denormalize_imgs(imgs):
    """将归一化的图像反归一化回[0, 255]范围"""
    imgs = (imgs + 1.0) / 2.0
    imgs = imgs * 255.0
    imgs = torch.clamp(imgs, 0, 255)
    return imgs

def save_batch(imgs, poses, yaws, output_dir, batch_idx=0):
    """
    保存batch中的图像和位姿信息
    
    Args:
        imgs: torch.Tensor, shape [batch_size, num_frames, num_cameras, 3, h, w] 或 [num_frames, num_cameras, 3, h, w]
        poses: torch.Tensor, shape [batch_size, num_frames, 2] 或 [num_frames, 2]
        yaws: torch.Tensor, shape [batch_size, num_frames, 1] 或 [num_frames, 1]
        output_dir: 输出目录
        batch_idx: batch索引
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理batch维度
    if imgs.dim() == 6:  # [batch_size, num_frames, num_cameras, 3, h, w]
        batch_size = imgs.shape[0]
        for b in range(batch_size):
            batch_dir = os.path.join(output_dir, f"batch_{batch_idx}_sample_{b}")
            os.makedirs(batch_dir, exist_ok=True)
            save_single_sample(imgs[b], poses[b], yaws[b], batch_dir)
    else:  # [num_frames, num_cameras, 3, h, w]
        save_single_sample(imgs, poses, yaws, output_dir)

def save_single_sample(imgs, poses, yaws, output_dir):
    """
    保存单个样本的16帧×6个摄像头的图像和位姿信息
    
    Args:
        imgs: torch.Tensor, shape [num_frames, num_cameras, 3, h, w] = [16, 6, 3, 256, 512]
        poses: torch.Tensor, shape [num_frames, 2] = [16, 2]
        yaws: torch.Tensor, shape [num_frames, 1] = [16, 1]
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 反归一化图像
    imgs_denorm = denormalize_imgs(imgs)  # [16, 6, 3, 256, 512]
    imgs_denorm = imgs_denorm.permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.uint8)  # [16, 6, 256, 512, 3]
    
    # 转换为numpy
    poses_np = poses.cpu().numpy()
    yaws_np = yaws.cpu().numpy()
    
    # 摄像头名称
    camera_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    # 保存每帧每个摄像头的图像
    for frame_idx in range(imgs_denorm.shape[0]):
        frame_dir = os.path.join(output_dir, f"frame_{frame_idx:02d}")
        os.makedirs(frame_dir, exist_ok=True)
        
        for cam_idx, cam_name in enumerate(camera_names):
            img = imgs_denorm[frame_idx, cam_idx]  # [256, 512, 3]
            # BGR转RGB（cv2保存需要BGR格式）
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_path = os.path.join(frame_dir, f"{cam_name}.jpg")
            cv2.imwrite(img_path, img_bgr)
        
        print(f"Saved frame {frame_idx:02d}: 6 camera images in {frame_dir}")
    
    # 保存位姿信息到JSON文件
    pose_info = {
        'num_frames': len(poses_np),
        'poses': poses_np.tolist(),  # [num_frames, 2] - 相对位置 (x, y)
        'yaws': yaws_np.tolist(),     # [num_frames, 1] - 相对yaw角（度）
    }
    
    json_path = os.path.join(output_dir, "poses.json")
    with open(json_path, 'w') as f:
        json.dump(pose_info, f, indent=2)
    print(f"Saved pose info: {json_path}")
    
    # 保存位姿信息到文本文件（便于查看）
    txt_path = os.path.join(output_dir, "poses.txt")
    with open(txt_path, 'w') as f:
        f.write("Frame\tX\tY\tYaw(deg)\n")
        for i in range(len(poses_np)):
            f.write(f"{i}\t{poses_np[i][0]:.6f}\t{poses_np[i][1]:.6f}\t{yaws_np[i][0]:.6f}\n")
    print(f"Saved pose text: {txt_path}")





"""
！！！生成 nuscenes 的 batch 数据！！！

------------------------------
O、预处理 epona 的 train.json
------------------------------

    这部分我已经做完了，可以直接看下一部分，此处仅记录实现细节。
    
    Epona Github 上给出的 train.json 中, 写入的场景有 700 个, val.json 中, 写入的场景有 150 个, 每个场景给出逐帧 ego_pose 和 FRONT 图片路径
    需要使用 epona_FRONT_2_epona_FULL_CAM.py 将扩张为六个视角, 即每个场景给出逐帧 ego_pose 和 六个视角的图片路径
    nuscenes 原文件里 samples 文件夹下存放的是数据集发布者设置的关键帧，有特殊标注, 每秒有2帧 samples; sweeps 文件夹下存放的是非关键帧，没有特殊标注
    所以最后的 train.json 中, samples 和 sweeps 文件夹是交替出现的

    另外需要注意的是 nuscenes 数据集中同一帧不同摄像机记录的 timestamp 不一样, 目前的 json 以 FRONT 的为准



------------------------------
一、Train/Val 的取帧逻辑
------------------------------

    nuscenes 原始fps=12
    这里 Train/Val 均采用 downsample_fps = 3, 即每隔4帧取1帧 (frame_idx = start + i * 4), 返回的16张图片是每秒钟3帧


    Train 固定从 0 开始取 (若想设置随机取, 直接注释掉 dataset.py 中的 157 行那附近就行)
    举例：
        假设序列有100帧, num_frames=16, downsample_fps=3
        则取该场景的第 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60 帧

    Val 多一个 target_frame 参数, 用于指定最后一帧是在该场景的第 target_frame 帧
    举例：
        假设序列有100帧, num_frames=16, downsample_fps=3, target_frame=-5 (等效于 target_frame=95)
        则取该场景的第 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95 帧




------------------------------
二、Batch 内的数据格式
------------------------------

    nuscenes 原数据里给的是世界坐标系下的绝对位置，这里返回的时候都改成某一帧相对于上一帧的相对位置了


     imgs (RGB 格式 256*512 图片, 6个摄像头):
     - 形状：[batch_size, 16, 6, 3, 256, 512]
     - 维度说明：[batch_size, 帧数, 摄像头数, RGB通道, 高度, 宽度]
     - 6个摄像头顺序：CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT
     - 数值范围：[-1.0, 1.0]（已归一化）
     - 用的归一化方法是 img = (img / 255.0 - 0.5) * 2


    poses (以米为单位, 相对于前一帧的位置变化量):
    - 形状：[batch_size, 16, 2]
    - 坐标系: X前进, Y右侧, Z向上 (NuScenes标准)
            - poses[:, t, 0]: 第t帧相对于第t-1帧的前进距离
            - poses[:, t, 1]: 第t帧相对于第t-1帧的右移距离
            - poses[:, 0, :] 没有意义, 没有第 -1 帧


    yaws (以°为单位, 相对于前一帧的方向角变化量):
    - 形状：[batch_size, 16, 1]
    - 正负: 向左转 (逆时针) 为正, 向右转 (顺时针) 为负,  [-180, 180] 度之间
            - yaws[:, t, 0]: 第t帧相对于第t-1帧的方向角变化量



"""
def main():
    print("Creating train dataset...")
    train_dataset = TrainDataset16Frames(
        nuscenes_path="/home/ma-user/modelarts/user-job-dir/jya/data/nuscenes/v1.0-trainval",
        epona_nuscenes_json_path="/home/ma-user/modelarts/user-job-dir/jya/code/MultiWorld/dataset/epona_nuscenes_train_FULL_CAM.json",
        num_frames=16,  # 固定16帧
        downsample_fps=3,
        downsample_size=16,
        h=256,
        w=512
    )
    print("Creating val dataset...")
    val_dataset = ValDataset16Frames(
        nuscenes_path="/home/ma-user/modelarts/user-job-dir/jya/data/nuscenes/v1.0-trainval",
        epona_nuscenes_json_path="/home/ma-user/modelarts/user-job-dir/jya/code/MultiWorld/dataset/epona_nuscenes_val_FULL_CAM.json",
        num_frames=16,  # 固定16帧
        downsample_fps=3,
        downsample_size=16,
        h=256,
        w=512,
        target_frame=-5  # 验证集的目标帧位置
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # 以 batch_size=2 创建 TrainDataLoader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0  # 设置为0避免多进程问题
    )
    
    # 以 batch_size=2 创建 DataLoader
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0  # 设置为0避免多进程问题
    )
    
    # 获取一个 batch 训练集，val 一样
    print("\nGetting a batch from train dataloader...")
    for batch_idx, (imgs, poses, yaws) in enumerate(train_dataloader):



        #######################################
        # 可以做其他训练逻辑，这里仅作简单保存
        #######################################




        print(f"\nTrain Batch {batch_idx}:")
        print(f"  Images shape: {imgs.shape}")   # [batch_size, 16, 6, 3, 256, 512]
        print(f"  Poses shape: {poses.shape}")   # [batch_size, 16, 2]
        print(f"  Yaws shape: {yaws.shape}")     # [batch_size, 16, 1]
        output_dir = "./batch_output_train"
        save_batch(imgs, poses, yaws, output_dir, batch_idx=batch_idx)


        # 示例，取一个 batch 就退出
        break
    
    print(f"\nTrain dataset test done! Batch saved to: {output_dir}")
    


if __name__ == "__main__":
    main()

