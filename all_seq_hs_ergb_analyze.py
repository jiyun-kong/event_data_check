import os
import numpy as np
import torch
from PIL import Image, ImageDraw
import glob
from pathlib import Path
from event_utils import events_to_voxel_torch

NUM_BINS = 3

# 제외할 이벤트 파일들 정의
EXCLUDED_EVENTS = {
    'far/bridge_lake_01': ['000000.npz'],
    'far/bridge_lake_03': ['000000.npz'],
    'far/lake_01': ['000000.npz'],
    'far/lake_03': ['000000.npz'],
    'close/fountain_schaffhauserplatz_02': ['000000.npz'],
    'close/spinning_umbrella': ['000000.npz'],
    'close/confetti': ['000014.npz', '000016.npz', '000018.npz', '000039.npz', '000070.npz', 
                       '000075.npz', '000082.npz', '000132.npz', '000170.npz', '000346.npz', '000349.npz'],
    'close/water_bomb_eth_01': ['002592.npz', '002620.npz', '002628.npz', '002639.npz', '002645.npz', 
                                '002649.npz', '002660.npz', '002664.npz', '002666.npz', '002668.npz', 
                                '002676.npz', '002677.npz', '002679.npz', '002680.npz', '002681.npz', 
                                '002683.npz', '002686.npz', '002687.npz', '002688.npz', '002693.npz', 
                                '002695.npz', '002699.npz', '002702.npz', '002703.npz', '002706.npz', 
                                '002711.npz', '002716.npz', '002717.npz', '002718.npz', '002775.npz', 
                                '002785.npz', '002793.npz', '002794.npz', '002797.npz', '002799.npz', 
                                '002808.npz', '002809.npz', '002815.npz', '002822.npz', '002823.npz', 
                                '002824.npz', '002825.npz', '002826.npz', '002827.npz', '002829.npz', 
                                '002831.npz', '002832.npz', '002834.npz', '002838.npz', '002841.npz', 
                                '002843.npz', '002844.npz', '002848.npz', '002855.npz', '002857.npz', 
                                '002862.npz', '002867.npz', '002871.npz', '002873.npz', '002877.npz', 
                                '002885.npz', '002887.npz', '002889.npz', '002894.npz', '002897.npz', 
                                '002901.npz', '002910.npz', '002927.npz', '002935.npz', '002936.npz', '002942.npz'],
    'close/water_bomb_floor_01': ['000066.npz', '000085.npz', '000092.npz', '000101.npz']
}

def analyze_sequence(sequence_path, base_path):
    """개별 시퀀스에 대해 모든 이벤트 파일의 voxel grid 분석을 수행합니다."""
    # 상대 경로 계산 (base_path 기준)
    rel_path = os.path.relpath(sequence_path, base_path)
    print(f"\n{rel_path}")
    print("-" * 50)
    
    try:
        # 이미지 경로 확인
        image_path = os.path.join(sequence_path, 'images_corrected')
        if not os.path.exists(image_path):
            print("  images_corrected 디렉토리가 없습니다.")
            return
            
        image_files = sorted(glob.glob(os.path.join(image_path, '*.png')))
        if not image_files:
            print("  이미지 파일이 없습니다.")
            return
            
        # 첫 번째 이미지로 크기 확인
        image = Image.open(image_files[0])
        print(f"  image shape: ({image.height}, {image.width})")
        
        # 이벤트 경로 확인
        event_path = os.path.join(sequence_path, 'events_aligned')
        if not os.path.exists(event_path):
            print("  events_aligned 디렉토리가 없습니다.")
            return
            
        event_files = sorted(glob.glob(os.path.join(event_path, '*.npz')))
        if not event_files:
            print("  이벤트 파일이 없습니다.")
            return
        
        # 제외할 이벤트 파일들 필터링
        excluded_files = EXCLUDED_EVENTS.get(rel_path, [])
        if excluded_files:
            print(f"  제외할 이벤트 파일들: {len(excluded_files)}개")
            event_files = [f for f in event_files if os.path.basename(f) not in excluded_files]
            print(f"  필터링 후 이벤트 파일 수: {len(event_files)}")
        
        if not event_files:
            print("  필터링 후 이벤트 파일이 없습니다.")
            return
        
        print(f"  총 이벤트 파일 수: {len(event_files)}")
        print()
        
        # 각 이벤트 파일에 대해 분석
        successful_files = 0
        failed_files = 0
        
        for i, event_file in enumerate(event_files):
            try:
                print(f"    파일 {i+1:3d}/{len(event_files)}: {os.path.basename(event_file)}")
                
                data = np.load(event_file)
                
                # 마스킹 적용 (이미지 범위 내 이벤트만 사용)
                FRAME_WIDTH = image.width
                FRAME_HEIGHT = image.height
                
                t = data['t']
                x_data = data['x']
                y_data = data['y']
                p = data['p']
                
                # 마스킹 조건 수정: < 대신 <= 사용하여 경계값 포함
                mask = (x_data < FRAME_WIDTH) & (y_data < FRAME_HEIGHT) & (x_data >= 0) & (y_data >= 0)
                t = t[mask]
                x_data = x_data[mask]
                y_data = y_data[mask]
                p = p[mask]
                
                # 이벤트 데이터 검증
                if len(t) == 0:
                    print(f"      경고: 마스킹 후 이벤트가 없습니다.")
                    failed_files += 1
                    continue
                
                dt = t.max() - t.min()
                if dt == 0:
                    print(f"      경고: dt = 0 (모든 이벤트가 동일한 타임스탬프)")
                    failed_files += 1
                    continue
                
                events_data = np.zeros((x_data.shape[0], 4), dtype=np.float32)  # [t, x, y, p]
                
                events_data[:, 0] = t
                events_data[:, 1] = x_data
                events_data[:, 2] = y_data
                events_data[:, 3] = np.where(p == 0, -1, 1) # 0 -> -1, 1 -> 1
                
                # event -> voxel grid
                event_vg = events_to_voxel_torch(events_data, num_bins=NUM_BINS, width=FRAME_WIDTH, height=FRAME_HEIGHT)
                
                # voxel grid 크기와 이미지 크기 비교
                expected_shape = (NUM_BINS, FRAME_HEIGHT, FRAME_WIDTH)
                actual_shape = event_vg.shape
                
                if actual_shape == expected_shape:
                    print(f"      성공: voxel shape {actual_shape} (이미지 크기와 일치)")
                    successful_files += 1
                else:
                    print(f"      크기 불일치: voxel shape {actual_shape}, 예상: {expected_shape}")
                    print(f"         이미지 크기: {FRAME_HEIGHT}x{FRAME_WIDTH}")
                    failed_files += 1
                
                
                
            except Exception as e:
                print(f"      오류: {str(e)}")
                failed_files += 1
            
            if i == 5:
                break
        
        # 시퀀스별 결과 요약
        print(f"\n{rel_path} 결과 요약:")
        print(f"    성공: {successful_files}개, 실패: {failed_files}개")
        if successful_files > 0:
            print(f"이 시퀀스는 voxel grid 생성이 가능합니다.")
        else:
            print(f"    이 시퀀스는 모든 이벤트 파일에서 voxel grid 생성에 실패했습니다.")
            
    except Exception as e:
        print(f"  오류 발생: {str(e)}")

def check_image_corruption(sequence_path, base_path):
    """개별 시퀀스의 이미지 파일 손상 여부를 확인합니다."""
    # 상대 경로 계산 (base_path 기준)
    rel_path = os.path.relpath(sequence_path, base_path)
    print(f"\n{rel_path}")
    print("-" * 50)
    
    try:
        # 이미지 경로 확인
        image_path = os.path.join(sequence_path, 'images_corrected')
        if not os.path.exists(image_path):
            print("  images_corrected 디렉토리가 없습니다.")
            return
            
        image_files = sorted(glob.glob(os.path.join(image_path, '*.png')))
        if not image_files:
            print("  이미지 파일이 없습니다.")
            return
        
        print(f"  총 이미지 파일 수: {len(image_files)}")
        print()
        
        # 각 이미지 파일에 대해 손상 여부 확인
        corrupted_files = []
        valid_files = 0
        
        for i, image_file in enumerate(image_files):
            try:
                # 이미지 파일 열기 시도
                with Image.open(image_file) as img:
                    # 이미지 로드 시도 (실제로 데이터를 읽어봄)
                    img.load()
                    
                    # 이미지 크기 확인
                    width, height = img.size
                    
                    # 첫 5개 파일만 상세 정보 출력
                    if i < 5:
                        print(f"    파일 {i+1:3d}/{len(image_files)}: {os.path.basename(image_file)} - 정상 ({width}x{height})")
                    
                    valid_files += 1
                    
            except Exception as e:
                error_msg = str(e)
                print(f"    파일 {i+1:3d}/{len(image_files)}: {os.path.basename(image_file)} - 손상됨")
                print(f"      오류: {error_msg}")
                corrupted_files.append((image_file, error_msg))
        
        # 시퀀스별 결과 요약
        print(f"\n{rel_path} 이미지 검사 결과:")
        print(f"    정상: {valid_files}개, 손상: {len(corrupted_files)}개")
        
        if corrupted_files:
            print(f"  손상된 파일들:")
            for file_path, error_msg in corrupted_files:
                print(f"    - {os.path.basename(file_path)}: {error_msg}")
        else:
            print(f"  모든 이미지 파일이 정상입니다.")
            
        return corrupted_files
            
    except Exception as e:
        print(f"  오류 발생: {str(e)}")
        return []

def check_all_sequences_images(base_path):
    """모든 시퀀스의 이미지 파일 손상 여부를 확인합니다."""
    print(f"기본 경로: {base_path}")
    print("=" * 60)
    
    # far와 close 디렉토리 확인
    far_path = os.path.join(base_path, 'far')
    close_path = os.path.join(base_path, 'close')
    
    all_corrupted_files = []
    
    if os.path.exists(far_path):
        print("\n[FAR 시퀀스들 이미지 검사]")
        far_sequences = [d for d in os.listdir(far_path) 
                        if os.path.isdir(os.path.join(far_path, d)) and not d.startswith('.')]
        
        for seq in sorted(far_sequences):
            seq_path = os.path.join(far_path, seq)
            corrupted = check_image_corruption(seq_path, base_path)
            all_corrupted_files.extend(corrupted)
    
    if os.path.exists(close_path):
        print("\n[CLOSE 시퀀스들 이미지 검사]")
        close_sequences = [d for d in os.listdir(close_path) 
                          if os.path.isdir(os.path.join(close_path, d)) and not d.startswith('.')]
        
        for seq in sorted(close_sequences):
            seq_path = os.path.join(close_path, seq)
            corrupted = check_image_corruption(seq_path, base_path)
            all_corrupted_files.extend(corrupted)
    
    # 전체 결과 요약
    print("\n" + "=" * 60)
    print("전체 이미지 검사 결과 요약")
    print("=" * 60)
    
    if all_corrupted_files:
        print(f"총 손상된 이미지 파일 수: {len(all_corrupted_files)}")
        print("\n손상된 파일 목록:")
        for file_path, error_msg in all_corrupted_files:
            print(f"  {file_path}: {error_msg}")
        
        # 손상된 파일 목록을 파일로 저장
        output_file = 'corrupted_images_hs_ergb.txt'
        with open(output_file, 'w') as f:
            f.write(f"# hs-ergb 데이터셋 손상된 이미지 파일 목록\n")
            f.write(f"# 검사 시간: {os.popen('date').read().strip()}\n\n")
            for file_path, error_msg in all_corrupted_files:
                f.write(f"{file_path}\t{error_msg}\n")
        print(f"\n손상된 파일 목록이 '{output_file}'에 저장되었습니다.")
        
    else:
        print("모든 이미지 파일이 정상입니다!")

def analyze_far_sequences(base_path):
    """far 디렉토리의 모든 시퀀스에 대해 분석을 수행합니다."""
    print(f"기본 경로: {base_path}")
    print("=" * 60)
    
    far_path = os.path.join(base_path, 'far')
    
    if not os.path.exists(far_path):
        print("far 디렉토리가 없습니다.")
        return
    
    print("\n[FAR 시퀀스들 분석]")
    print("제외할 이벤트 파일들을 필터링하여 분석합니다.")
    
    far_sequences = [d for d in os.listdir(far_path) 
                    if os.path.isdir(os.path.join(far_path, d)) and not d.startswith('.')]
    
    for seq in sorted(far_sequences):
        seq_path = os.path.join(far_path, seq)
        analyze_sequence(seq_path, base_path)

def main():
    base_path = '/scratch2/jiyun.kong/hs_ergb'
    check_all_sequences_images(base_path)

if __name__ == "__main__":
    main()