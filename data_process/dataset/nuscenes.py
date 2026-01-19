"""
conda create -n drive --clone PyTorch-2.1.0
conda activate drive
pip install einops  scipy==1.12.0 imageio pillow opencv-python==4.9.0.80  numpy==1.23.0
"""
import json
import cv2
import os
import numpy as np
import torch
import random
from torch.utils.data import DataLoader, Dataset
from scipy.spatial.transform import Rotation as R
from einops import rearrange
import imageio
import math
from PIL import Image



def reverse_seq_data(poses, seqs):
    seq_len = len(poses)

    reverse_seq = seqs[::-1]
    start_pose = poses.pop(-1)
    reverse_poses = [-pi for pi in poses]
    reverse_poses = [start_pose, ] + reverse_poses[::-1]

    return reverse_poses, reverse_seq


def get_meta_data(poses):
    poses = np.concatenate([poses[0:1], poses], axis=0)
    rel_pose = np.linalg.inv(poses[:-1]) @ poses[1:]
    # rel_pose=  np.concatenate([rel_pose], axis=0)
    xyzs = rel_pose[:, :3, 3]
    xys = xyzs[:, :2]
    
    
    def radians_to_degrees(radians):
        degrees = radians * (180 / math.pi)
        return degrees

    rel_yaws = radians_to_degrees(R.from_matrix(rel_pose[:,:3,:3]).as_euler('zyx', degrees=False)[:,0])[:, np.newaxis]

    # rel_poses_yaws=np.concatenate([xys,rel_yaws[:,None]],axis=1)
    return {
        'rel_poses': xys,
        'rel_yaws': rel_yaws,
        # 'rel_poses_xyz': xyzs,
        # 'rel_poses_yaws':rel_poses_yaws,
    }


def quaternion_to_rotation_matrix(q):
    q_w, q_x, q_y, q_z = q
    
    R = np.array([
        [1 - 2*(q_y**2 + q_z**2), 2*(q_x*q_y - q_w*q_z), 2*(q_x*q_z + q_w*q_y)],
        [2*(q_x*q_y + q_w*q_z), 1 - 2*(q_x**2 + q_z**2), 2*(q_y*q_z - q_w*q_x)],
        [2*(q_x*q_z - q_w*q_y), 2*(q_y*q_z + q_w*q_x), 1 - 2*(q_x**2 + q_y**2)]
    ])
    return R

def create_transformation_matrix(rotation, translation):
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T

class TrainDataset16Frames(Dataset):
    """
    修改后的数据集类，每次返回16张图像和对应位姿
    基于原始的TrainImgDataset，但修改为固定返回16张图像
    支持6个摄像头：CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT
    """
    def __init__(
        self, nuscenes_path, epona_nuscenes_json_path, num_frames=16, downsample_fps=3, 
        downsample_size=16, h=256, w=512, reverse_seq=False
    ):
        self.img_path_data = []
        self.pose_data = []
        self.nuscenes_path = nuscenes_path
        self.num_frames = num_frames  # 固定为16张图像
        # 定义6个摄像头
        self.camera_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        with open(epona_nuscenes_json_path, 'r', encoding='utf-8') as file:
            nuscenes_preprocess_data = json.load(file)
        self.ori_fps = 12  # 10 hz
        self.downsample = self.ori_fps // downsample_fps
        nuscenes_keys = sorted(list(nuscenes_preprocess_data.keys()))
        for video_keys in nuscenes_keys:
            tmp_camera_paths = []  # 存储每帧的6个摄像头路径
            tmp_pose = []
            img_path_poses = nuscenes_preprocess_data[video_keys]
            # 确保序列长度足够
            if len(img_path_poses) < self.num_frames * self.downsample:
                continue
            for img_path_pose in img_path_poses:
                # 从JSON中读取6个摄像头的路径
                camera_paths_dict = {
                    'CAM_FRONT': img_path_pose.get('front_data_path', ''),
                    'CAM_FRONT_LEFT': img_path_pose.get('front_left_data_path', ''),
                    'CAM_FRONT_RIGHT': img_path_pose.get('front_right_data_path', ''),
                    'CAM_BACK': img_path_pose.get('back_data_path', ''),
                    'CAM_BACK_LEFT': img_path_pose.get('back_left_data_path', ''),
                    'CAM_BACK_RIGHT': img_path_pose.get('back_right_data_path', '')
                }
                # 转换为完整路径
                camera_paths_full = {}
                for cam_name, rel_path in camera_paths_dict.items():
                    if rel_path:
                        camera_paths_full[cam_name] = os.path.join(nuscenes_path, rel_path)
                    else:
                        camera_paths_full[cam_name] = None
                tmp_camera_paths.append(camera_paths_full)
                tmp_pose.append(img_path_pose['ego_pose'])
            self.img_path_data.append(tmp_camera_paths)
            self.pose_data.append(tmp_pose)
        self.h = h
        self.w = w
        self.downsample_size = downsample_size
        self.reverse_seq = reverse_seq
        print(f"Dataset initialized: num_frames={self.num_frames}, downsample={self.downsample}, "
              f"downsample_fps={downsample_fps}, total sequences={len(self.img_path_data)}")

    def __len__(self):
        return len(self.img_path_data)    
    
    def aug_seq(self, imgs):
        """裁剪图像序列到固定尺寸"""
        ih, iw, _ = imgs[0].shape
        assert self.h == 256 and self.w == 512, f"Expected h=256, w=512, got h={self.h}, w={self.w}"
        if iw == 512:
            x = int(ih/2 - self.h/2)
            y = 0
        else:
            x = 0
            y = int(iw/2 - self.w/2)
        for i in range(len(imgs)):
            imgs[i] = imgs[i][x:x+self.h, y:y+self.w, :]
        return imgs   
    
    def normalize_imgs(self, imgs):
        """归一化图像到[-1, 1]范围"""
        imgs = imgs / 255.0
        imgs = (imgs - 0.5) * 2
        return imgs


    def getimg(self, index):
        """获取16帧×6个摄像头的图像和对应位姿"""
        camera_paths_list = self.img_path_data[index]  # 每帧包含6个摄像头路径的字典
        poses = self.pose_data[index]
        clip_length = len(camera_paths_list)
        
        start = 0
        # # 随机选择起始位置，确保能取到16张图像
        # max_start = clip_length - self.num_frames * self.downsample
        # if max_start <= 0:
        #     # 如果序列太短，从头开始，但可能无法取到16张
        #     start = 0
        # else:
        #     start = random.randint(0, max_start)
        
        # ims格式: [16帧, 6摄像头, H, W, 3]
        ims = []
        poses_new = []
        
        for i in range(self.num_frames):
            frame_idx = start + i * self.downsample
            if frame_idx >= clip_length:
                # 如果超出范围，重复最后一张
                frame_idx = clip_length - 1
            
            # 从JSON中直接获取6个摄像头的路径（已经是完整路径）
            camera_paths = camera_paths_list[frame_idx]
            
            # 加载6个摄像头的图像
            frame_cameras = []
            for cam_name in self.camera_names:
                cam_full_path = camera_paths.get(cam_name)
                
                try:
                    if cam_full_path and os.path.exists(cam_full_path):
                        im = cv2.cvtColor(cv2.imread(cam_full_path), cv2.COLOR_BGR2RGB)
                        if im is None:
                            raise ValueError(f"Failed to read image: {cam_full_path}")
                    else:
                        # 如果路径为空或文件不存在，使用黑色图像
                        if not cam_full_path:
                            raise FileNotFoundError(f"Camera {cam_name} path is empty")
                        else:
                            raise FileNotFoundError(f"Camera image not found: {cam_full_path}")
                except Exception as e:
                    print(f"Error reading camera {cam_name} image: {e}")
                    # 如果读取失败，使用黑色图像
                    im = np.zeros((self.h, self.w, 3), dtype=np.uint8)
                
                # Resize图像
                h, w, _ = im.shape
                if 2*h < w:
                    w_1 = round(w / h * self.h)
                    im = cv2.resize(im, (w_1, self.h))
                else:
                    h_1 = round(h / w * self.w)
                    im = cv2.resize(im, (self.w, h_1))
                
                frame_cameras.append(im)
            
            # frame_cameras: [6, H, W, 3]
            ims.append(frame_cameras)
            
            # 将位姿字典转换为4x4变换矩阵
            pose_dict = poses[frame_idx]
            if isinstance(pose_dict, dict):
                # 假设位姿格式为 {'rotation': [w, x, y, z], 'translation': [x, y, z]}
                rotation = quaternion_to_rotation_matrix(pose_dict['rotation'])
                translation = pose_dict['translation']
                pose_matrix = create_transformation_matrix(rotation, translation)
                poses_new.append(pose_matrix)
            else:
                # 如果已经是矩阵格式，直接使用
                poses_new.append(pose_dict)
        
        # ims格式: [16, 6, H, W, 3]
        # 将poses_new转换为numpy数组
        poses_array = np.array(poses_new)
        # 计算相对位姿和yaw角
        pose_dict = get_meta_data(poses=poses_array)
        return ims, pose_dict['rel_poses'], pose_dict['rel_yaws']
            
    def __getitem__(self, index):
        """返回16帧×6个摄像头的图像和对应位姿
        
        Returns:
            imgs: [16, 6, 3, 256, 512] - 16帧，每帧6个摄像头，RGB通道，256x512
            poses: [16, 2] - 相对位置
            yaws: [16, 1] - 相对yaw角
        """
        imgs, poses, yaws = self.getimg(index)
        
        # imgs格式: [16, 6, H, W, 3]
        # 对每一帧的每个摄像头进行aug_seq和normalize
        imgs_processed = []
        for frame_idx in range(len(imgs)):
            frame_cameras = imgs[frame_idx]  # [6, H, W, 3]
            frame_cameras_processed = []
            for cam_idx in range(len(frame_cameras)):
                cam_img = frame_cameras[cam_idx]  # [H, W, 3]
                # aug_seq需要处理图像列表，对单个图像也适用
                cam_img_list = [cam_img]  # 转为列表格式
                cam_img_aug = self.aug_seq(cam_img_list)[0]  # 裁剪到固定尺寸
                frame_cameras_processed.append(cam_img_aug)
            imgs_processed.append(frame_cameras_processed)
        
        # 转换为tensor
        imgs_tensor = []
        poses_tensor = []
        yaws_tensor = []
        
        for frame_idx, (frame_cameras, pose, yaw) in enumerate(zip(imgs_processed, poses, yaws)):
            frame_tensors = []
            for cam_img in frame_cameras:
                # cam_img: [H, W, 3] -> [3, H, W]
                frame_tensors.append(torch.from_numpy(cam_img.copy()).permute(2, 0, 1))
            # frame_tensors: [6, 3, H, W]
            imgs_tensor.append(torch.stack(frame_tensors, 0))  # [6, 3, H, W]
            poses_tensor.append(torch.from_numpy(pose.copy()))
            yaws_tensor.append(torch.from_numpy(yaw.copy()))
        
        # Stack所有帧
        # imgs_tensor: [16, 6, 3, H, W]
        imgs = torch.stack(imgs_tensor, 0)  # [16, 6, 3, H, W]
        imgs = self.normalize_imgs(imgs)  # 归一化到[-1, 1]
        
        return imgs, torch.stack(poses_tensor, 0).float(), torch.stack(yaws_tensor, 0).float()


class ValDataset16Frames(Dataset):
    """
    验证数据集，固定返回16帧×6个摄像头
    """
    def __init__(
        self, nuscenes_path, epona_nuscenes_json_path, num_frames=16, downsample_fps=3, 
        downsample_size=16, h=256, w=512, target_frame=-5
    ):
        self.img_path_data = []
        self.pose_data = []
        self.target_frame = target_frame
        assert self.target_frame < 0
        self.nuscenes_path = nuscenes_path
        self.num_frames = num_frames
        # 定义6个摄像头
        self.camera_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        with open(epona_nuscenes_json_path, 'r', encoding='utf-8') as file:
            nuscenes_preprocess_data = json.load(file)
        self.ori_fps = 12
        self.downsample = self.ori_fps // downsample_fps
        nuscenes_keys = sorted(list(nuscenes_preprocess_data.keys()))
        for video_keys in nuscenes_keys:
            tmp_camera_paths = []  # 存储每帧的6个摄像头路径
            tmp_pose = []
            img_path_poses = nuscenes_preprocess_data[video_keys]
            if len(img_path_poses) < self.num_frames * self.downsample:
                continue
            for img_path_pose in img_path_poses:
                # 从JSON中读取6个摄像头的路径
                camera_paths_dict = {
                    'CAM_FRONT': img_path_pose.get('front_data_path', ''),
                    'CAM_FRONT_LEFT': img_path_pose.get('front_left_data_path', ''),
                    'CAM_FRONT_RIGHT': img_path_pose.get('front_right_data_path', ''),
                    'CAM_BACK': img_path_pose.get('back_data_path', ''),
                    'CAM_BACK_LEFT': img_path_pose.get('back_left_data_path', ''),
                    'CAM_BACK_RIGHT': img_path_pose.get('back_right_data_path', '')
                }
                # 转换为完整路径
                camera_paths_full = {}
                for cam_name, rel_path in camera_paths_dict.items():
                    if rel_path:
                        camera_paths_full[cam_name] = os.path.join(nuscenes_path, rel_path)
                    else:
                        camera_paths_full[cam_name] = None
                tmp_camera_paths.append(camera_paths_full)
                tmp_pose.append(img_path_pose['ego_pose'])
            self.img_path_data.append(tmp_camera_paths)
            self.pose_data.append(tmp_pose)
        self.h = h
        self.w = w
        self.downsample_size = downsample_size
        print(f"ValDataset initialized: num_frames={self.num_frames}, downsample={self.downsample}, "
              f"total sequences={len(self.img_path_data)}")

    def __len__(self):
        return len(self.img_path_data)    
    
    def aug_seq(self, imgs):
        ih, iw, _ = imgs[0].shape
        assert self.h == 256 and self.w == 512
        if iw == 512:
            x = int(ih/2 - self.h/2)
            y = 0
        else:
            x = 0
            y = int(iw/2 - self.w/2)
        for i in range(len(imgs)):
            imgs[i] = imgs[i][x:x+self.h, y:y+self.w, :]
        return imgs   
    
    def normalize_imgs(self, imgs):
        imgs = imgs / 255.0
        imgs = (imgs - 0.5) * 2
        return imgs

    def getimg(self, index):
        """获取16帧×6个摄像头的图像和对应位姿（验证集）"""
        camera_paths_list = self.img_path_data[index]  # 每帧包含6个摄像头路径的字典
        poses = self.pose_data[index]
        clip_length = len(camera_paths_list)
        
        target_index = clip_length + self.target_frame
        start_index = target_index - self.downsample * self.num_frames
        
        if start_index < 0:
            start_index = 0
        
        ims = []
        poses_new = []
        for i in range(self.num_frames):
            frame_idx = start_index + i * self.downsample
            if frame_idx >= clip_length:
                frame_idx = clip_length - 1
            
            # 从JSON中直接获取6个摄像头的路径（已经是完整路径）
            camera_paths = camera_paths_list[frame_idx]
            
            # 加载6个摄像头的图像
            frame_cameras = []
            for cam_name in self.camera_names:
                cam_full_path = camera_paths.get(cam_name)
                
                try:
                    if cam_full_path and os.path.exists(cam_full_path):
                        im = cv2.cvtColor(cv2.imread(cam_full_path), cv2.COLOR_BGR2RGB)
                        if im is None:
                            raise ValueError(f"Failed to read image: {cam_full_path}")
                    else:
                        # 如果路径为空或文件不存在，使用黑色图像
                        if not cam_full_path:
                            raise FileNotFoundError(f"Camera {cam_name} path is empty")
                        else:
                            raise FileNotFoundError(f"Camera image not found: {cam_full_path}")
                except Exception as e:
                    print(f"Error reading camera {cam_name} image: {e}")
                    im = np.zeros((self.h, self.w, 3), dtype=np.uint8)
                
                h, w, _ = im.shape
                if 2*h < w:
                    w_1 = round(w / h * self.h)
                    im = cv2.resize(im, (w_1, self.h))
                else:
                    h_1 = round(h / w * self.w)
                    im = cv2.resize(im, (self.w, h_1))
                
                frame_cameras.append(im)
            
            ims.append(frame_cameras)
            
            # 将位姿字典转换为4x4变换矩阵
            pose_dict = poses[frame_idx]
            if isinstance(pose_dict, dict):
                rotation = quaternion_to_rotation_matrix(pose_dict['rotation'])
                translation = pose_dict['translation']
                pose_matrix = create_transformation_matrix(rotation, translation)
                poses_new.append(pose_matrix)
            else:
                poses_new.append(pose_dict)
        
        poses_array = np.array(poses_new)
        pose_dict = get_meta_data(poses=poses_array)
        return ims, pose_dict['rel_poses'], pose_dict['rel_yaws']
            
    def __getitem__(self, index):
        """返回16帧×6个摄像头的图像和对应位姿（与TrainDataset16Frames相同）"""
        imgs, poses, yaws = self.getimg(index)
        
        # imgs格式: [16, 6, H, W, 3]
        imgs_processed = []
        for frame_idx in range(len(imgs)):
            frame_cameras = imgs[frame_idx]  # [6, H, W, 3]
            frame_cameras_processed = []
            for cam_idx in range(len(frame_cameras)):
                cam_img = frame_cameras[cam_idx]  # [H, W, 3]
                cam_img_list = [cam_img]
                cam_img_aug = self.aug_seq(cam_img_list)[0]
                frame_cameras_processed.append(cam_img_aug)
            imgs_processed.append(frame_cameras_processed)
        
        imgs_tensor = []
        poses_tensor = []
        yaws_tensor = []
        
        for frame_idx, (frame_cameras, pose, yaw) in enumerate(zip(imgs_processed, poses, yaws)):
            frame_tensors = []
            for cam_img in frame_cameras:
                frame_tensors.append(torch.from_numpy(cam_img.copy()).permute(2, 0, 1))
            imgs_tensor.append(torch.stack(frame_tensors, 0))
            poses_tensor.append(torch.from_numpy(pose.copy()))
            yaws_tensor.append(torch.from_numpy(yaw.copy()))
        
        imgs = torch.stack(imgs_tensor, 0)  # [16, 6, 3, H, W]
        imgs = self.normalize_imgs(imgs)
        
        return imgs, torch.stack(poses_tensor, 0).float(), torch.stack(yaws_tensor, 0).float()

