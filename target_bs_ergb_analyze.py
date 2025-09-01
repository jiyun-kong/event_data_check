import os
import numpy as np
from PIL import Image, ImageDraw
import glob
from pathlib import Path
from event_utils import extract_events, events_to_voxel_grid, convert_and_fix_event_pixels

FRAME_WIDTH = 970
FRAME_HEIGHT = 625
NUM_BINS = 10

def analyze_sequence(sequence_path, base_path):
    """개별 시퀀스에 대해 모든 이벤트 파일의 bins를 분석합니다."""
    # 상대 경로 계산 (base_path 기준)
    rel_path = os.path.relpath(sequence_path, base_path)
    print(f"\n{rel_path}")
    print("-" * 50)
    
    try:
        # 이미지 경로 확인
        image_path = os.path.join(sequence_path, 'images')
        if not os.path.exists(image_path):
            print("  images 디렉토리가 없습니다.")
            return
            
        image_files = sorted(glob.glob(os.path.join(image_path, '*.png')))
        if not image_files:
            print("  이미지 파일이 없습니다.")
            return
            
        # 첫 번째 이미지로 크기 확인
        image = Image.open(image_files[0])
        print(f"  image shape: ({image.height}, {image.width})")
        
        # 이벤트 경로 확인
        event_path = os.path.join(sequence_path, 'events')
        if not os.path.exists(event_path):
            print("  events 디렉토리가 없습니다.")
            return
            
        event_files = sorted(glob.glob(os.path.join(event_path, '*.npz')))
        if not event_files:
            print("  이벤트 파일이 없습니다.")
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
                
                # 이벤트 데이터 준비
                events_data = np.zeros((data['x'].shape[0], 4), dtype=np.float32)  # [t, x, y, p]
                
                x_data = convert_and_fix_event_pixels(data['x'], FRAME_WIDTH - 1)
                y_data = convert_and_fix_event_pixels(data['y'], FRAME_HEIGHT - 1)
                p = data['polarity']
                
                events_data[:, 0] = data['timestamp']
                events_data[:, 1] = x_data
                events_data[:, 2] = y_data
                events_data[:, 3] = p
                
                # 이벤트 데이터 상세 정보 출력 (첫 3개 파일만)
                if i < 3:
                    print(f"      원본 이벤트 수: {data['x'].shape[0]}")
                    print(f"      마스킹 후 이벤트 수: {len(events_data)}")
                    print(f"      타임스탬프 범위: {data['timestamp'].min():.6f} ~ {data['timestamp'].max():.6f}")
                    print(f"      X 좌표 범위: {x_data.min()} ~ {x_data.max()}")
                    print(f"      Y 좌표 범위: {y_data.min()} ~ {y_data.max()}")
                    print(f"      극성 값: {np.unique(p)}")
                
                # 이벤트 데이터 검증
                if len(events_data) == 0:
                    print(f"      경고: 이벤트가 없습니다.")
                    failed_files += 1
                    continue
                
                # 이벤트 수가 너무 적은 경우 체크 (최소 2개 이상 필요)
                if len(events_data) < 2:
                    print(f"      경고: 이벤트가 너무 적습니다 (필요: 2개 이상, 현재: {len(events_data)}개)")
                    failed_files += 1
                    continue
                
                # timestamp 검증
                t = events_data[:, 0]
                dt = t.max() - t.min()
                if dt == 0:
                    print(f"      경고: dt = 0 (모든 이벤트가 동일한 타임스탬프)")
                    failed_files += 1
                    continue
                
                # event -> voxel grid
                event_vg = events_to_voxel_grid(events_data, num_bins=NUM_BINS, width=FRAME_WIDTH, height=FRAME_HEIGHT)
                
                # voxel grid 크기와 이미지 크기 비교
                expected_shape = (NUM_BINS, FRAME_HEIGHT, FRAME_WIDTH)
                actual_shape = event_vg.shape
                
                if actual_shape == expected_shape:
                    print(f"      성공: voxel shape {actual_shape} (이미지 크기와 일치)")
                    
                    # 각 bin의 min/max 값 출력 (첫 3개 파일만)
                    if i < 3:
                        for j in range(NUM_BINS):
                            bin_data = event_vg[j]
                            print(f"        Bin {j}: min={bin_data.min():.4f}, max={bin_data.max():.4f}")
                    
                    successful_files += 1
                else:
                    print(f"      크기 불일치: voxel shape {actual_shape}, 예상: {expected_shape}")
                    print(f"         이미지 크기: {FRAME_HEIGHT}x{FRAME_WIDTH}")
                    failed_files += 1
                
            except Exception as e:
                print(f"      오류: {str(e)}")
                failed_files += 1
        
        # 시퀀스별 결과 요약
        print(f"\n{rel_path} 결과 요약:")
        print(f"    성공: {successful_files}개, 실패: {failed_files}개")
        if successful_files > 0:
            print(f"이 시퀀스는 voxel grid 생성이 가능합니다.")
        else:
            print(f"    이 시퀀스는 모든 이벤트 파일에서 voxel grid 생성에 실패했습니다.")
            
    except Exception as e:
        print(f"  오류 발생: {str(e)}")

def analyze_all_sequences(base_path):
    """모든 시퀀스에 대해 분석을 수행합니다."""
    print(f"기본 경로: {base_path}")
    print("=" * 60)
    
    if not os.path.exists(base_path):
        print("기본 경로가 존재하지 않습니다.")
        return
    
    print("\n[모든 시퀀스 분석]")
    print("각 시퀀스의 모든 이벤트 파일에 대해 bins를 확인합니다.")
    
    sequences = [d for d in os.listdir(base_path) 
                if os.path.isdir(os.path.join(base_path, d)) and not d.startswith('.')]

    for seq in sorted(sequences):
        seq_path = os.path.join(base_path, seq)
        analyze_sequence(seq_path, base_path)
    
    # 전체 결과 요약
    print("\n" + "=" * 60)
    print("전체 결과 요약")
    print("=" * 60)
    print(f"총 시퀀스 수: {len(sequences)}")

def main():
    base_path = '/scratch2/jiyun.kong/bs_ergb/test'
    analyze_all_sequences(base_path)

if __name__ == "__main__":
    main()