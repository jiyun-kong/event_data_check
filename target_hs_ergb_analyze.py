import os
import numpy as np
import torch
from PIL import Image, ImageDraw
import glob
from pathlib import Path
from event_utils import events_to_voxel_torch

NUM_BINS = 3

def events_vg_visualize(base_path):
    base_output_path = 'baloon_popping_vis_results'
    os.makedirs(base_output_path, exist_ok=True)

    image_path = os.path.join(base_path, 'images_corrected')
    image_files = sorted(glob.glob(os.path.join(image_path, '*.png')))
    image = image_files[1]

    event_path = os.path.join(base_path, 'events_aligned')
    event_files = sorted(glob.glob(os.path.join(event_path, '*.npz')))

    # 배경으로 깔아둘 이미지
    image = Image.open(image).convert('RGBA')
    image_array = np.array(image)
    image_array[:, :, 3] = int(255 * 0.3)
    image = Image.fromarray(image_array)


    print(f"event file path : {event_files[3]}")
    data = np.load(event_files[3])

    # 마스킹 적용 (이미지 범위 내 이벤트만 사용)
    FRAME_WIDTH = image.width
    FRAME_HEIGHT = image.height

    t = data['t']
    x_data = data['x']
    y_data = data['y']
    p = data['p']

    mask = (x_data <= FRAME_WIDTH-1) & (y_data <= FRAME_HEIGHT-1) & (x_data >= 0) & (y_data >= 0)
    t = t[mask]
    x_data = x_data[mask]
    y_data = y_data[mask]
    p = p[mask]

    events_data = np.zeros((x_data.shape[0], 4), dtype=np.float32)  # [t, x, y, p]

    events_data[:, 0] = t
    events_data[:, 1] = x_data
    events_data[:, 2] = y_data
    events_data[:, 3] = np.where(p == 0, -1, 1) # 0 -> -1, 1 -> 1
    print(f"events_data : {events_data.shape}")
    print(f"Frame Height : {FRAME_HEIGHT}, Frame Width : {FRAME_WIDTH}\n")
    
    # event -> voxel grid
    event_vg = events_to_voxel_torch(events_data, num_bins=NUM_BINS, width=FRAME_WIDTH, height=FRAME_HEIGHT)
    
    for i in range(NUM_BINS):
        print(f"Bin {i}: min={event_vg[i].min()}, max={event_vg[i].max()}")
        
        vis_image = Image.new('RGBA', (FRAME_WIDTH, FRAME_HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(vis_image)

        # voxel grid의 절댓값 최대값 계산 (투명도 스케일링용)
        max_abs_value = torch.max(torch.abs(event_vg[i])).item()

        H, W = event_vg[i].shape
        for y in range(H):
            for x in range(W):
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
    base_path = '/scratch2/jiyun.kong/hs_ergb/close/baloon_popping'
    print(f"기본 경로: {base_path}")

    events_vg_visualize(base_path)


if __name__ == "__main__":
    main()