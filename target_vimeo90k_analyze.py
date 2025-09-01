import os
import numpy as np
from PIL import Image, ImageDraw
import glob
from pathlib import Path
from event_utils import events_to_voxel_grid_vimeo90k

FRAME_WIDTH = 448
FRAME_HEIGHT = 256
NUM_BINS = 5

def events_vg_visualize(base_path):
    base_output_path = '00005_0002_vis_results'
    os.makedirs(base_output_path, exist_ok=True)

    image_path = os.path.join(base_path, 'imgs')
    image_files = sorted(glob.glob(os.path.join(image_path, '*.png')))
    
    event_file = os.path.join(base_path, 'events.npz')
    timestamps_file = os.path.join(base_path, 'timestamps.txt')

    print(f"event file path : {event_file}")
    print(f"timestamps file path : {timestamps_file}")
    
    # timestamps.txt 파일 로드
    timestamps = np.loadtxt(timestamps_file)
    data = np.load(event_file)

    print(f"총 이미지 개수: {len(image_files)}")
    print(f"총 이벤트 개수: {len(data['x'])}")

    # 각 이미지에 대해 해당 시간 구간의 이벤트만 처리 (마지막 이미지 제외)
    for img_idx in range(len(image_files) - 1):
        print(f"\n=== 이미지 {img_idx} 처리 중 ===")
        
        # i번째 이미지의 시작 시간과 끝 시간
        if img_idx == 0:
            start_t = 0.0
            end_t = timestamps[1]
        else:
            start_t = timestamps[img_idx]
            end_t = timestamps[img_idx+1]
        
        print(f"시간구간: {start_t:.6f} ~ {end_t:.6f}")
        
        # 해당 시간 구간의 이벤트만 필터링
        time_mask = (data['t'] >= start_t) & (data['t'] < end_t)
        
        # 필터링된 이벤트 데이터 생성
        filtered_events = np.zeros((np.sum(time_mask), 4), dtype=np.float32)  # [t, x, y, p]
        filtered_events[:, 0] = data['t'][time_mask]
        filtered_events[:, 1] = data['x'][time_mask]
        filtered_events[:, 2] = data['y'][time_mask]
        filtered_events[:, 3] = data['p'][time_mask]
        
        print(f"필터링된 이벤트 수: {len(filtered_events)}")
        print(f"  - 시간 범위: {filtered_events[:, 0].min():.6f} ~ {filtered_events[:, 0].max():.6f}")
        
        # 배경으로 깔아둘 이미지 (해당 인덱스의 이미지 사용)
        image = Image.open(image_files[img_idx]).convert('RGBA')
        image_array = np.array(image)
        image_array[:, :, 3] = int(255 * 0.3)
        image = Image.fromarray(image_array)

        # event -> voxel grid
        event_vg = events_to_voxel_grid_vimeo90k(filtered_events, num_bins=NUM_BINS, width=FRAME_WIDTH, height=FRAME_HEIGHT)
        print(f"event_vg.shape : {event_vg.shape}")
        
        # 각 bin의 시간 구간 계산
        filtered_times = filtered_events[:, 0]
        time_min, time_max = filtered_times.min(), filtered_times.max()
        time_delta = time_max - time_min
        bin_time_step = time_delta / NUM_BINS
        
        print(f"  - 이벤트 시간 범위: {time_min:.6f} ~ {time_max:.6f}")
        print(f"  - 시간 델타: {time_delta:.6f}")
        print(f"  - 각 bin 시간 간격: {bin_time_step:.6f}")
        
        for bin_idx in range(NUM_BINS):
            bin_start = time_min + bin_idx * bin_time_step
            bin_end = time_min + (bin_idx + 1) * bin_time_step
            print(f"  - Bin {bin_idx}: {bin_start:.6f} ~ {bin_end:.6f} (min={event_vg[bin_idx].min()}, max={event_vg[bin_idx].max()})")
            
            vis_image = Image.new('RGBA', (FRAME_WIDTH, FRAME_HEIGHT), (0, 0, 0, 0))
            draw = ImageDraw.Draw(vis_image)

            # voxel grid의 절댓값 최대값 계산 (투명도 스케일링용)
            max_abs_value = np.max(np.abs(event_vg[bin_idx]))
            
            if max_abs_value > 0:  # 이벤트가 있는 경우에만 처리
                for y in range(FRAME_HEIGHT):
                    for x in range(FRAME_WIDTH):
                        value = event_vg[bin_idx][y, x]

                        if value != 0:
                            alpha_scale = 0.1 + 0.9 * (abs(value) / max_abs_value)
                            alpha = int(255 * alpha_scale)

                            if value > 0: # positive
                                color = (255, 0, 0, alpha)
                            else:
                                color = (0, 0, 255, alpha)
                            
                            # 점 그리기 (크기 2x2)
                            draw.rectangle([x-1, y-1, x+1, y+1], fill=color)

            # 이미지 합성
            result_image = Image.alpha_composite(vis_image, image)
            
            # 파일명에 이미지 인덱스와 bin 인덱스 포함
            output_filename = f'img_{img_idx:04d}_bin_{bin_idx}.png'
            output_path = os.path.join(base_output_path, output_filename)
            result_image.save(output_path, 'PNG')

            print(f"  bin {bin_idx} saved : {output_filename}")
    

    
def main():
    base_path = '/scratch2/jiyun.kong/vimeo90k/00005/0002'
    print(f"기본 경로: {base_path}")

    events_vg_visualize(base_path)


if __name__ == "__main__":
    main()