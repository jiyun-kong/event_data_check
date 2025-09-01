#!/usr/bin/env python3
"""
Vimeo90k 데이터셋에서 망가진 events.npz 파일을 찾는 스크립트

사용법:
    python vimeo_events_npz_check.py /scratch2/jiyun.kong/vimeo90k_split/train
"""

import os
import sys
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('npz_check.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_npz_file(npz_path):
    """
    단일 npz 파일을 검사하여 유효성 확인
    
    Args:
        npz_path (str): npz 파일 경로
        
    Returns:
        dict: 검사 결과 딕셔너리
    """
    result = {
        'path': npz_path,
        'is_valid': False,
        'error': None,
        'file_size': 0,
        'keys': [],
        'shapes': {},
        'dtypes': {}
    }
    
    try:
        # 파일 존재 여부 확인
        if not os.path.exists(npz_path):
            result['error'] = 'File does not exist'
            return result
            
        # 파일 크기 확인
        result['file_size'] = os.path.getsize(npz_path)
        if result['file_size'] == 0:
            result['error'] = 'Empty file (0 bytes)'
            return result
            
        # npz 파일 로드 시도
        with np.load(npz_path, allow_pickle=False) as data:
            result['keys'] = list(data.keys())
            
            # 각 키의 데이터 형태와 타입 확인
            for key in data.keys():
                array = data[key]
                result['shapes'][key] = array.shape
                result['dtypes'][key] = str(array.dtype)
                
                # 기본적인 데이터 유효성 검사
                if array.size == 0:
                    result['error'] = f'Empty array for key: {key}'
                    return result
                    
                # NaN 또는 무한대 값 확인
                if np.issubdtype(array.dtype, np.floating):
                    if np.any(np.isnan(array)) or np.any(np.isinf(array)):
                        result['error'] = f'NaN or Inf values found in key: {key}'
                        return result
        
        result['is_valid'] = True
        
    except Exception as e:
        result['error'] = f'Exception: {str(e)}'
        
    return result

def scan_directory(base_path, max_workers=4):
    """
    디렉토리를 스캔하여 모든 npz 파일을 찾고 검사
    
    Args:
        base_path (str): 검사할 기본 디렉토리 경로
        max_workers (int): 병렬 처리를 위한 최대 워커 수
        
    Returns:
        tuple: (유효한 파일 목록, 망가진 파일 목록)
    """
    logger.info(f"디렉토리 스캔 시작: {base_path}")
    
    # 모든 npz 파일 경로 수집
    npz_files = []
    base_path = Path(base_path)
    
    # train/00001/0001/events.npz 패턴으로 검색
    for group_dir in base_path.iterdir():
        if group_dir.is_dir() and group_dir.name.isdigit():
            for seq_dir in group_dir.iterdir():
                if seq_dir.is_dir() and seq_dir.name.isdigit():
                    npz_file = seq_dir / "events.npz"
                    if npz_file.exists():
                        npz_files.append(str(npz_file))
    
    logger.info(f"총 {len(npz_files)}개의 npz 파일을 발견했습니다.")
    
    if not npz_files:
        logger.warning("npz 파일을 찾을 수 없습니다.")
        return [], []
    
    # 병렬로 파일 검사
    valid_files = []
    corrupted_files = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 작업 제출
        future_to_path = {executor.submit(check_npz_file, npz_path): npz_path 
                         for npz_path in npz_files}
        
        # 진행 상황 표시와 함께 결과 수집
        for future in tqdm(as_completed(future_to_path), total=len(npz_files), 
                          desc="NPZ 파일 검사 중"):
            result = future.result()
            
            if result['is_valid']:
                valid_files.append(result)
            else:
                corrupted_files.append(result)
                # 망가진 파일 즉시 로깅
                rel_path = os.path.relpath(result['path'], base_path)
                logger.warning(f"망가진 파일 발견: {rel_path} - {result['error']}")
    
    return valid_files, corrupted_files

def generate_report(valid_files, corrupted_files, base_path):
    """
    검사 결과 보고서 생성
    
    Args:
        valid_files (list): 유효한 파일 목록
        corrupted_files (list): 망가진 파일 목록
        base_path (str): 기본 디렉토리 경로
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"npz_check_report_{timestamp}.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Vimeo90k Events NPZ 파일 검사 보고서\n")
        f.write("=" * 80 + "\n")
        f.write(f"검사 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"검사 경로: {base_path}\n")
        f.write(f"총 파일 수: {len(valid_files) + len(corrupted_files)}\n")
        f.write(f"유효한 파일: {len(valid_files)}\n")
        f.write(f"망가진 파일: {len(corrupted_files)}\n")
        f.write("\n")
        
        if corrupted_files:
            f.write("망가진 파일 목록:\n")
            f.write("-" * 80 + "\n")
            
            # 그룹별로 정리
            corrupted_by_group = {}
            for file_info in corrupted_files:
                rel_path = os.path.relpath(file_info['path'], base_path)
                parts = rel_path.split(os.sep)
                if len(parts) >= 2:
                    group = parts[0]
                    seq = parts[1]
                    if group not in corrupted_by_group:
                        corrupted_by_group[group] = []
                    corrupted_by_group[group].append((seq, file_info))
            
            for group in sorted(corrupted_by_group.keys()):
                f.write(f"\n그룹 {group}:\n")
                for seq, file_info in sorted(corrupted_by_group[group]):
                    f.write(f"  - {seq}: {file_info['error']} (파일 크기: {file_info['file_size']} bytes)\n")
        else:
            f.write("모든 파일이 유효합니다! 🎉\n")
        
        # 통계 정보
        if valid_files:
            f.write("\n" + "=" * 80 + "\n")
            f.write("유효한 파일 통계:\n")
            f.write("-" * 80 + "\n")
            
            total_size = sum(file_info['file_size'] for file_info in valid_files)
            avg_size = total_size / len(valid_files)
            f.write(f"총 크기: {total_size / (1024**2):.2f} MB\n")
            f.write(f"평균 크기: {avg_size / 1024:.2f} KB\n")
            
            # 키 정보 (첫 번째 유효한 파일 기준)
            if valid_files[0]['keys']:
                f.write(f"NPZ 키 목록: {', '.join(valid_files[0]['keys'])}\n")
                for key in valid_files[0]['keys']:
                    shape = valid_files[0]['shapes'][key]
                    dtype = valid_files[0]['dtypes'][key]
                    f.write(f"  - {key}: shape={shape}, dtype={dtype}\n")
    
    logger.info(f"보고서가 생성되었습니다: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Vimeo90k events.npz 파일 무결성 검사')
    parser.add_argument('data_path', help='검사할 데이터 디렉토리 경로')
    parser.add_argument('--workers', type=int, default=4, help='병렬 처리 워커 수 (기본값: 4)')
    parser.add_argument('--verbose', '-v', action='store_true', help='상세한 로그 출력')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # 경로 유효성 확인
    if not os.path.exists(args.data_path):
        logger.error(f"경로가 존재하지 않습니다: {args.data_path}")
        sys.exit(1)
    
    if not os.path.isdir(args.data_path):
        logger.error(f"디렉토리가 아닙니다: {args.data_path}")
        sys.exit(1)
    
    try:
        # 파일 검사 실행
        valid_files, corrupted_files = scan_directory(args.data_path, args.workers)
        
        # 결과 출력
        logger.info("=" * 60)
        logger.info("검사 완료!")
        logger.info(f"총 파일 수: {len(valid_files) + len(corrupted_files)}")
        logger.info(f"유효한 파일: {len(valid_files)}")
        logger.info(f"망가진 파일: {len(corrupted_files)}")
        
        if corrupted_files:
            logger.warning("\n망가진 파일이 발견되었습니다:")
            for file_info in corrupted_files:
                rel_path = os.path.relpath(file_info['path'], args.data_path)
                logger.warning(f"  - {rel_path}: {file_info['error']}")
        else:
            logger.info("모든 파일이 유효합니다! 🎉")
        
        # 보고서 생성
        generate_report(valid_files, corrupted_files, args.data_path)
        
        # 종료 코드 설정 (망가진 파일이 있으면 1, 없으면 0)
        sys.exit(1 if corrupted_files else 0)
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"예상치 못한 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()