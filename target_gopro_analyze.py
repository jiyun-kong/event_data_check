import os
import numpy as np
from PIL import Image, ImageDraw
import glob
from pathlib import Path
from event_utils import read_h5_events, read_h5_image, events_to_voxel_grid, convert_and_fix_event_pixels

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
NUM_BINS = 10

def events_vg_visualize(base_path):
    seq_name = os.path.basename(base_path)
    h5_file = os.path.join(base_path, f'{seq_name}.h5')

    base_output_path = f'{seq_name}_vis_results'
    os.makedirs(base_output_path, exist_ok=True)

    events = read_h5_events(h5_file)
    x = events[:, 0]
    y = events[:, 1]
    t = events[:, 2]
    p = events[:, 3]

    images = read_h5_image(h5_file)

    # 0번째 이미지에 대해서만 voxel grid 생성
    i = 0
    image = Image.fromarray(images[i][..., [2, 1, 0]]).convert('RGBA')
    image_array = np.array(image)
    image_array[:, :, 3] = int(255 * 0.3)
    image = Image.fromarray(image_array)

    
    t0 = t.min()
    t1 = t.max()
    total_duration = t1 - t0
    num_frames = len(images)
    
    exposure_time = total_duration / (num_frames - 1)

    # i번째 프레임 ~ (i+1)번째 프레임 구간
    start_t = t0 + i * exposure_time
    end_t = t0 + (i + 1) * exposure_time

    mask = (t >= start_t) & (t < end_t)
    t_sel = t[mask]
    x_sel = x[mask]
    y_sel = y[mask]
    p_sel = p[mask]

    events_sel = np.stack([t_sel, x_sel, y_sel, p_sel], axis=1)
    print(f"[frame {i}] 이벤트 수: {len(x_sel)}, 시간구간: {start_t:.1f} ~ {end_t:.1f}")


    # event -> voxel grid
    event_vg = events_to_voxel_grid(events_sel, num_bins=NUM_BINS, width=FRAME_WIDTH, height=FRAME_HEIGHT)
    print(f"event_vg.shape : {event_vg.shape}")

    

    for i in range(NUM_BINS):
        print(f"Bin {i}: min={event_vg[i].min()}, max={event_vg[i].max()}")
        
        vis_image = Image.new('RGBA', (FRAME_WIDTH, FRAME_HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(vis_image)

        # voxel grid의 절댓값 최대값 계산 (투명도 스케일링용)
        max_abs_value = np.max(np.abs(event_vg[i]))

        for y in range(FRAME_HEIGHT):
            for x in range(FRAME_WIDTH):
                value = event_vg[i][y, x]

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
        output_path = os.path.join(base_output_path, f'bin_{i}.png')
        result_image.save(output_path, 'PNG')

        print(f"bin {i} saved : {output_path}")
    

    
def main():
    base_path = '/scratch2/jiyun.kong/gopro/test/GOPR0881_11_01'
    print(f"기본 경로: {base_path}")

    events_vg_visualize(base_path)


if __name__ == "__main__":
    main()