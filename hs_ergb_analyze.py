import os
import numpy as np
from PIL import Image, ImageDraw
import glob
from pathlib import Path
from event_utils import extract_events, convert_and_fix_event_pixels

def image_analyze(base_path):
    image_path = os.path.join(base_path, 'images_corrected')
    image_files = sorted(glob.glob(os.path.join(image_path, '*.png')))

    print("\n총 이미지 개수 : ", len(image_files))

    try:
        if image_files:
            print(f"image file path : {image_files[0]}")
            
            first_image = image_files[0]
            with Image.open(first_image) as img:
                print(f"size : {img.size}")
                print(f"width : {img.size[0]}")
                print(f"height : {img.size[1]}")
                print(f"dtype : {img.format}")
    except Exception as e:
        print(f"event 데이터 에러 ({os.path.basename(image_path)}): {str(e)}")

def events_npz(base_path):
    event_path = os.path.join(base_path, 'events_aligned')
    event_files = sorted(glob.glob(os.path.join(event_path, '*.npz')))

    print("\n총 이벤트 개수 : ", len(event_files))

    # 기본적인 이벤트 정보
    try:
        for i in range(2):
            print(f"event file path : {event_files[i]}")
            data = np.load(event_files[i])
            
            events_stack = extract_events(data, dataset='hs_ergb')
            print(f"events_stack.shape : {events_stack.shape}")

            
            for key in data.keys():
                arr = data[key]

                if key == 't':
                    if arr.size > 0:
                        print(f"=== Timestamp")
                        print(f"- shape : {arr.shape}")
                        print(f"- dtype : {arr.dtype}")
                        print(f"- min : {float(arr.min())}")
                        print(f"- max : {float(arr.max())}")
                        print(f"- duration : {float(arr.max() - arr.min())}")
                        
                elif key == 'x':
                    if arr.size > 0:
                        print(f"=== x")
                        print(f"- shape : {arr.shape}")
                        print(f"- dtype : {arr.dtype}")
                        print(f"- min : {float(arr.min())}")
                        print(f"- max : {float(arr.max())}")

                elif key == 'y':
                    if arr.size > 0:
                        print(f"=== y")
                        print(f"- shape : {arr.shape}")
                        print(f"- dtype : {arr.dtype}")
                        print(f"- min : {float(arr.min())}")
                        print(f"- max : {float(arr.max())}")


                elif key == 'p':
                    if arr.size > 0:
                        print(f"=== polarity")
                        print(f"- shape : {arr.shape}")
                        print(f"- dtype : {arr.dtype}")
                        print(f"- min : {float(arr.min())}")
                        print(f"- max : {float(arr.max())}")

    except Exception as e:
        print(f"event 데이터 에러 ({os.path.basename(event_path)}): {str(e)}")


def image_event_align(base_path):
    seq_name = base_path.split('/')[-2]

    image_path = os.path.join(base_path, 'images_corrected')
    event_path = os.path.join(base_path, 'events_aligned')
    output_path = 'vis_results_hs_ergb'
    os.makedirs(output_path, exist_ok=True)

    image_files = glob.glob(os.path.join(image_path, '*.png'))
    event_files = glob.glob(os.path.join(event_path, '*.npz'))

    image_0 = image_files[0]
    image_1 = image_files[1]
    event = event_files[0]

    # 이미지 0
    image_0 = Image.open(image_0).convert('RGBA')
    image_array_0 = np.array(image_0)
    image_array_0[:, :, 3] = int(255 * 0.3)
    image_0 = Image.fromarray(image_array_0)
    W, H = image_0.size  # 이미지 크기 추가

    # 이미지 1
    image_1 = Image.open(image_1).convert('RGBA')
    image_array_1 = np.array(image_1)
    image_array_1[:, :, 3] = int(255 * 0.3)
    image_1 = Image.fromarray(image_array_1)

    event = np.load(event)

    # 투명 배경 생성
    vis_image = Image.new('RGBA', image_0.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(vis_image)

    # x, y 좌표 수정
    x_data = event['x']
    y_data = event['y']
    p = event['p']

    # 마스킹 적용 (이미지 범위 내 이벤트만 사용)
    mask = (x_data <= W-1) & (y_data <= H-1) & (x_data >= 0) & (y_data >= 0)
    x_data = x_data[mask]
    y_data = y_data[mask]
    p = p[mask]

    # 이벤트 그리기
    for i in range(len(x_data)):
        x = x_data[i]
        y = y_data[i]
        polarity = p[i]

        if polarity == 1:
            color = (255, 0, 0, 255)
        else:
            color = (0, 0, 255, 255)

        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)


    result_image_0 = Image.alpha_composite(vis_image, image_0)
    output_path_0 = os.path.join(output_path, f'{seq_name}_image_event_align_0.png')
    result_image_0.save(output_path_0, 'PNG')

    result_image_1 = Image.alpha_composite(vis_image, image_1)
    output_path_1 = os.path.join(output_path, f'{seq_name}_image_event_align_1.png')
    result_image_1.save(output_path_1, 'PNG')

    print(f"image_event_align saved : {output_path_0}")
    print(f"image_event_align saved : {output_path_1}")

def main():
    base_path = '/scratch2/jiyun.kong/hs_ergb'
    print(f"기본 경로: {base_path}")
    
    far_close_ts = ['far', 'close']

    for fc in far_close_ts:
        base_fc_path = os.path.join(base_path, fc)
        seq_paths = glob.glob(os.path.join(base_fc_path, '*/'))

        for seq_path in seq_paths:
            print(f"\n\n시퀀스 경로: {seq_path}")

            image_analyze(seq_path)
            events_npz(seq_path)
            image_event_align(seq_path)


if __name__ == "__main__":
    main()