import os
import numpy as np
import torch
import glob
from pathlib import Path
from event_utils import events_to_voxel_torch

NUM_BINS = 3

def check_dt_zero_events(base_path):
    """dt = 0인 이벤트 파일들을 찾아냅니다."""
    print(f"검사 경로: {base_path}")
    print("=" * 60)
    
    event_path = os.path.join(base_path, 'events_aligned')
    event_files = sorted(glob.glob(os.path.join(event_path, '*.npz')))
    
    print(f"총 이벤트 파일 수: {len(event_files)}")
    print("-" * 60)

    dt_zero_files = []

    for i, event_file in enumerate(event_files):
        try:
            data = np.load(event_file)
            t = data['t']
            
            # 타임스탬프 정보
            t_min = t.min()
            t_max = t.max()
            dt = t_max - t_min
            unique_timestamps = len(np.unique(t))
            
            print(f"파일 {i+1:3d}: {os.path.basename(event_file)}")
            print(f"  타임스탬프 범위: {t_min:.6f} ~ {t_max:.6f}")
            print(f"  dt: {dt:.6f}")
            print(f"  고유 타임스탬프 수: {unique_timestamps}")
            
            if dt == 0:
                print(f"경고: dt = 0 (모든 이벤트가 동일한 타임스탬프)")
                dt_zero_files.append((event_file, t_min, unique_timestamps))
            else:
                print(f"정상: dt > 0")
            
            print()
            
        except Exception as e:
            print(f"파일 {i+1:3d}: {os.path.basename(event_file)} - 오류 발생: {str(e)}")
            print()

    return dt_zero_files
    
    

def main():
    base_path = '/scratch2/jiyun.kong/hs_ergb/close/water_bomb_floor_01'
    dt_zero_files = check_dt_zero_events(base_path)

    print(dt_zero_files)
    
    

if __name__ == "__main__":
    main()