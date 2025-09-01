import os
import numpy as np
from PIL import Image, ImageDraw
import glob
from pathlib import Path


def image_analyze(base_path):
    image_path = os.path.join(base_path, 'imgs')
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
    
    return image_files

def events_npz(base_path):
    timestamp_file = os.path.join(base_path, 'timestamps.txt')
    event_file = os.path.join(base_path, 'events.npz')

    # 기본적인 이벤트 정보
    try:
        print(f"event file path : {event_file}")
        timestamps = np.loadtxt(timestamp_file)
        data = np.load(event_file)
        
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
        print(f"event 데이터 에러 ({os.path.basename(event_file)}): {str(e)}")
    
    return timestamps, data

def image_event_align(base_path, timestamps, events, image_files):
    output_path = 'vimeo90k_vis_result'
    os.makedirs(output_path, exist_ok=True)
    
    # 첫 번째 이미지로부터 크기 정보 가져오기
    with Image.open(image_files[0]) as first_img:
        FRAME_WIDTH, FRAME_HEIGHT = first_img.size
    
    # 이벤트 데이터 구조 확인 및 정리
    # events.npz에서 't', 'x', 'y', 'p' 키를 가진 데이터를 가져옴
    t = events['t']  # timestamp
    x = events['x']  # x 좌표
    y = events['y']  # y 좌표
    p = events['p']  # polarity
    
    print(f"총 이벤트 수: {len(t)}")
    print(f"이미지 개수: {len(image_files)}")
    print(f"타임스탬프 개수: {len(timestamps)}")
    
    # 각 이미지에 대해 해당 시간 구간의 이벤트만 필터링
    for i in range(len(image_files)):
        # i번째 이미지의 시작 시간과 끝 시간
        if i == 0:
            # 첫 번째 이미지: 0.0 ~ timestamps[0]
            start_t = 0.0
            end_t = timestamps[0]
        elif i == len(image_files) - 1:
            # 마지막 이미지: timestamps[i-1] ~ 마지막 이벤트 시간
            start_t = timestamps[i-1]
            end_t = t.max()
        else:
            # 중간 이미지들: timestamps[i-1] ~ timestamps[i]
            start_t = timestamps[i-1]
            end_t = timestamps[i]
        
        print(f"[이미지 {i}] 시간구간: {start_t:.6f} ~ {end_t:.6f}")
        
        # 해당 시간 구간의 이벤트만 필터링
        mask = (t >= start_t) & (t < end_t)
        x_sel = x[mask]
        y_sel = y[mask]
        p_sel = p[mask]
        t_sel = t[mask]
        
        print(f"  - 이벤트 수: {len(x_sel)}")
        
        # 이미지 로드
        with Image.open(image_files[i]) as image:
            image = image.convert('RGBA')
            image_array = np.array(image)
            image_array[:, :, 3] = int(255 * 0.3)  # 투명도 설정
            image = Image.fromarray(image_array)
        
        # 투명 배경 생성
        vis_image = Image.new('RGBA', (FRAME_WIDTH, FRAME_HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(vis_image)
        
        # 이벤트 그리기
        for xi, yi, pi in zip(x_sel, y_sel, p_sel):
            color = (255, 0, 0, 255) if pi == 1 else (0, 0, 255, 255)
            draw.ellipse((xi - 2, yi - 2, xi + 2, yi + 2), fill=color)
        
        # 이미지 합성
        result_image = Image.alpha_composite(vis_image, image)
        
        # 파일명에서 시퀀스 이름 추출 (시퀀스/세부시퀀스 형태)
        seq_name = os.path.basename(os.path.dirname(base_path))  # 00005
        sub_seq_name = os.path.basename(base_path)  # 0002
        seq_full_name = f'{seq_name}_{sub_seq_name}'  # 00005_0002
        output_filename = f'{seq_full_name}_{i:04d}_image_event_align.png'
        output_path_full = os.path.join(output_path, output_filename)
        
        result_image.save(output_path_full, 'PNG')
        print(f"  - 저장됨: {output_filename}")


def main():
    base_path = '/scratch2/jiyun.kong/vimeo90k/00005/0002'
    print(f"기본 경로: {base_path}")

    image_files = image_analyze(base_path)
    timestamps, events = events_npz(base_path)
    image_event_align(base_path, timestamps, events, image_files)


if __name__ == "__main__":
    main()