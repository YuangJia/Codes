"""
将仅有FRONT摄像机的JSON文件转换为包含6个摄像机的JSON文件


1. 读取 epona_nuscenes_train_FRONT.json（包含1110个场景）
2. 对于每个场景的每一帧，从data_path提取文件名
3. 在sample_data.json中查找该文件名对应的条目，获取sample_token
4. 在sample_data.json中查找所有具有相同sample_token的条目
5. 筛选出6个摄像机的条目，提取filename和timestamp
6. 将6个摄像机的路径写入新的JSON文件


python epona_FRONT_2_epona_FULL_CAM.py
"""

import json
import os
from tqdm import tqdm
from collections import defaultdict

# 配置路径
FRONT_JSON_PATH = "/home/ma-user/modelarts/user-job-dir/jya/code/MultiWorld/dataset/epona_nuscenes_train_FRONT.json"
SAMPLE_DATA_JSON_PATH = "/home/ma-user/modelarts/user-job-dir/jya/data/nuscenes/v1.0-trainval/v1.0-trainval/sample_data.json"
OUTPUT_JSON_PATH = "/home/ma-user/modelarts/user-job-dir/jya/code/MultiWorld/dataset/epona_nuscenes_train_FULL_CAM.json"

# 6个摄像机的名称映射
CAMERA_NAMES = {
    'CAM_FRONT': 'front',
    'CAM_FRONT_LEFT': 'front_left',
    'CAM_FRONT_RIGHT': 'front_right',
    'CAM_BACK': 'back',
    'CAM_BACK_LEFT': 'back_left',
    'CAM_BACK_RIGHT': 'back_right'
}

# 反向映射：从输出字段名到摄像机名称
OUTPUT_FIELD_TO_CAMERA = {
    'front_data_path': 'CAM_FRONT',
    'front_left_data_path': 'CAM_FRONT_LEFT',
    'front_right_data_path': 'CAM_FRONT_RIGHT',
    'back_data_path': 'CAM_BACK',
    'back_left_data_path': 'CAM_BACK_LEFT',
    'back_right_data_path': 'CAM_BACK_RIGHT'
}


def load_sample_data(sample_data_path):
    """
    加载sample_data.json并建立索引
    
    Returns:
        filename_to_entry: 从filename到条目的映射
        sample_token_to_entries: 从sample_token到条目列表的映射
    """
    print(f"Loading sample_data.json from {sample_data_path}...")
    with open(sample_data_path, 'r', encoding='utf-8') as f:
        sample_data = json.load(f)
    
    print(f"Total entries in sample_data.json: {len(sample_data)}")
    
    # 建立两个索引
    filename_to_entry = {}  # filename -> entry
    sample_token_to_entries = defaultdict(list)  # sample_token -> [entries]
    
    for entry in sample_data:
        filename = entry.get('filename', '')
        sample_token = entry.get('sample_token', '')
        
        if filename:
            filename_to_entry[filename] = entry
        
        if sample_token:
            sample_token_to_entries[sample_token].append(entry)
    
    print(f"Indexed {len(filename_to_entry)} entries by filename")
    print(f"Indexed {len(sample_token_to_entries)} unique sample_tokens")
    
    return filename_to_entry, sample_token_to_entries


def is_camera_entry(entry):
    """判断一个条目是否是摄像机数据"""
    filename = entry.get('filename', '')
    # 检查filename中是否包含CAM_前缀
    return 'CAM_' in filename and filename.endswith('.jpg')


def get_camera_name_from_filename(filename):
    """从filename中提取摄像机名称"""
    # 格式: samples/CAM_FRONT/... 或 sweeps/CAM_FRONT/...
    parts = filename.split('/')
    for part in parts:
        if part.startswith('CAM_'):
            return part
    return None


def convert_frame_data(frame_data, filename_to_entry, sample_token_to_entries, verbose=False):
    """
    转换单帧数据，从FRONT路径找到其他5个摄像机的路径
    
    Args:
        frame_data: 原始帧数据，包含data_path和ego_pose
        filename_to_entry: filename到条目的映射
        sample_token_to_entries: sample_token到条目列表的映射
        verbose: 是否打印详细信息
    
    Returns:
        转换后的帧数据，包含6个摄像机的路径
    """
    # 获取FRONT摄像机的data_path
    front_data_path = frame_data.get('data_path', '')
    if not front_data_path:
        if verbose:
            print(f"Warning: No data_path in frame_data")
        return None
    
    # 在sample_data.json中查找该文件名对应的条目
    front_entry = filename_to_entry.get(front_data_path)
    if not front_entry:
        if verbose:
            print(f"Warning: Cannot find entry for {front_data_path}")
        return None
    
    # 获取sample_token
    sample_token = front_entry.get('sample_token', '')
    if not sample_token:
        if verbose:
            print(f"Warning: No sample_token found for {front_data_path}")
        return None
    
    # 查找所有具有相同sample_token的条目
    all_entries = sample_token_to_entries.get(sample_token, [])
    
    if verbose:
        print(f"Found {len(all_entries)} entries with sample_token {sample_token}")
    
    # 筛选出6个摄像机的条目
    camera_entries = {}
    for entry in all_entries:
        if is_camera_entry(entry):
            filename = entry.get('filename', '')
            camera_name = get_camera_name_from_filename(filename)
            if camera_name and camera_name in CAMERA_NAMES:
                camera_entries[camera_name] = {
                    'filename': filename,
                    'timestamp': entry.get('timestamp', 0)
                }
    
    # 检查是否找到了所有6个摄像机
    missing_cameras = set(CAMERA_NAMES.keys()) - set(camera_entries.keys())
    if missing_cameras and verbose:
        print(f"Warning: Missing cameras {missing_cameras} for sample_token {sample_token}")
    
    # 构建输出数据
    output_frame = {
        'timestamp': frame_data.get('timestamp', front_entry.get('timestamp', 0)),
        'ego_pose': frame_data.get('ego_pose', {})
    }
    
    # 添加6个摄像机的路径
    for output_field, camera_name in OUTPUT_FIELD_TO_CAMERA.items():
        if camera_name in camera_entries:
            output_frame[output_field] = camera_entries[camera_name]['filename']
        else:
            # 如果找不到该摄像机的数据，使用空字符串
            output_frame[output_field] = ""
            if verbose:
                print(f"Warning: Cannot find {camera_name} for sample_token {sample_token}")
    
    return output_frame


def convert_json(front_json_path, sample_data_path, output_json_path):
    """
    主转换函数
    
    Args:
        front_json_path: 输入的FRONT JSON文件路径
        sample_data_path: sample_data.json文件路径
        output_json_path: 输出的完整JSON文件路径
    """
    print("=" * 80)
    print("Converting FRONT-only JSON to FULL CAMERA JSON")
    print("=" * 80)
    
    # 加载sample_data.json并建立索引
    filename_to_entry, sample_token_to_entries = load_sample_data(sample_data_path)
    
    # 加载FRONT JSON文件
    print(f"\nLoading FRONT JSON from {front_json_path}...")
    with open(front_json_path, 'r', encoding='utf-8') as f:
        front_data = json.load(f)
    
    print(f"Total scenes in FRONT JSON: {len(front_data)}")
    
    # 转换每个场景
    output_data = {}
    total_frames = 0
    success_frames = 0
    failed_frames = 0
    missing_camera_frames = 0  # 统计缺少某些摄像机的帧数
    
    for scene_name, scene_frames in tqdm(front_data.items(), desc="Processing scenes"):
        output_frames = []
        
        for frame_idx, frame_data in enumerate(tqdm(scene_frames, desc=f"  Processing {scene_name}", leave=False)):
            total_frames += 1
            converted_frame = convert_frame_data(
                frame_data, filename_to_entry, sample_token_to_entries, verbose=False
            )
            
            if converted_frame:
                # 检查是否所有6个摄像机都找到了
                missing_count = sum(1 for field in OUTPUT_FIELD_TO_CAMERA.keys() 
                                  if not converted_frame.get(field))
                if missing_count > 0:
                    missing_camera_frames += 1
                
                output_frames.append(converted_frame)
                success_frames += 1
            else:
                failed_frames += 1
                if failed_frames <= 10:  # 只打印前10个失败信息
                    print(f"Failed to convert frame {frame_idx} in {scene_name}")
        
        if output_frames:
            output_data[scene_name] = output_frames
            if len(output_frames) < len(scene_frames):
                print(f"Scene {scene_name}: {len(output_frames)}/{len(scene_frames)} frames converted")
    
    # 保存输出JSON文件
    print(f"\nSaving output JSON to {output_json_path}...")
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("Conversion Summary")
    print("=" * 80)
    print(f"Total scenes: {len(output_data)}")
    print(f"Total frames processed: {total_frames}")
    print(f"Successfully converted: {success_frames}")
    print(f"Failed: {failed_frames}")
    print(f"Frames with missing cameras: {missing_camera_frames}")
    if total_frames > 0:
        print(f"Success rate: {success_frames/total_frames*100:.2f}%")
    print(f"Output saved to: {output_json_path}")
    print("=" * 80)


def main():
    """主函数"""
    # 检查输入文件是否存在
    if not os.path.exists(FRONT_JSON_PATH):
        print(f"Error: FRONT JSON file not found: {FRONT_JSON_PATH}")
        return
    
    if not os.path.exists(SAMPLE_DATA_JSON_PATH):
        print(f"Error: sample_data.json file not found: {SAMPLE_DATA_JSON_PATH}")
        return
    
    # 执行转换
    convert_json(FRONT_JSON_PATH, SAMPLE_DATA_JSON_PATH, OUTPUT_JSON_PATH)


if __name__ == "__main__":
    main()

