import os
import numpy as np
from PIL import Image, ImageDraw
import glob
from pathlib import Path
from event_utils import read_h5_events, read_h5_image


def event_analyze(seq_path):
    # seq_name = os.path.basename(seq_path)
    # event_file = os.path.join(seq_path, seq_name + '.h5')
    events = read_h5_events(seq_path)

    return events

def image_analyze(seq_path):
    # seq_name = os.path.basename(seq_path)
    # event_file = os.path.join(seq_path, seq_name + '.h5')

    images = read_h5_image(seq_path)

    return images

def image_event_align(images, events, seq_path):
    print(f"images[0].shape : {images[0].shape}")
    print(f"events.shape : {events.shape}")

    seq_name = os.path.basename(seq_path)
    print(f"seq_name : {seq_name}")

    output_path = 'gopro_vis_result'
    os.makedirs(output_path, exist_ok=True)
    FRAME_WIDTH, FRAME_HEIGHT = 1280, 720

    x = events[:, 0]
    y = events[:, 1]
    t = events[:, 2]
    p = events[:, 3]

    t0 = t.min()
    t1 = t.max()
    total_duration = t1 - t0
    num_frames = len(images)
    print(f"num_frames : {num_frames}")

    
    exposure_time = total_duration / (num_frames - 1)
    print(f"t.min(): {t0}")
    print(f"t.max(): {t1}")
    print(f"Duration (us): {total_duration}")
    print(f"Exposure per frame (us): {exposure_time}")

    for i in range(num_frames - 1):
        image = Image.fromarray(images[i][..., [2, 1, 0]]).convert('RGBA')
        image_array = np.array(image)
        image_array[:, :, 3] = int(255 * 0.3)
        image = Image.fromarray(image_array)

        # i번째 프레임 ~ (i+1)번째 프레임 구간
        start_t = t0 + i * exposure_time
        end_t = t0 + (i + 1) * exposure_time

        mask = (t >= start_t) & (t < end_t)
        x_sel = x[mask]
        y_sel = y[mask]
        p_sel = p[mask]

        print(f"[frame {i}] 이벤트 수: {len(x_sel)}, 시간구간: {start_t:.1f} ~ {end_t:.1f}")

        # # 투명 배경 생성
        # vis_image = Image.new('RGBA', (FRAME_WIDTH, FRAME_HEIGHT), (0, 0, 0, 0))
        # draw = ImageDraw.Draw(vis_image)

        # # 이벤트 그리기
        # for xi, yi, pi in zip(x_sel, y_sel, p_sel):
        #     color = (255, 0, 0, 255) if pi == 1 else (0, 0, 255, 255)

        #     draw.ellipse((xi - 2, yi - 2, xi + 2, yi + 2), fill=color)

        # # 이미지 합성
        # result_image = Image.alpha_composite(vis_image, image)
        # output_path_0 = os.path.join(output_path, f'{seq_name}_{i}_image_event_align.png')
        # result_image.save(output_path_0, 'PNG')



def main():
    base_path = '/scratch2/jiyun.kong/gopro'
    print(f"기본 경로: {base_path}")
    
    # tr_ts = ['train', 'test']
    tr_ts = ['test']

    for tt in tr_ts:
        base_tt_path = os.path.join(base_path, tt)
        seq_paths = glob.glob(os.path.join(base_tt_path, '*'))

        for seq_path in seq_paths:
            print(f"\n시퀀스 경로: {seq_path}")

            images = image_analyze(seq_path)
            events = event_analyze(seq_path)
            image_event_align(images, events, seq_path)

            return

            



if __name__ == "__main__":
    main()