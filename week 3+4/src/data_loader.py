# -*- coding: utf-8 -*-
"""
Module: data_loader.py
Dự án: SMART LMS & GENERATIVE AI
Chức năng: Đọc và tải các tệp dữ liệu OULAD từ thư mục raw.
"""

import os
import pandas as pd
from typing import Dict, Optional, Generator

def load_table(file_path: str, na_values=None, chunksize: Optional[int] = None):
    """Đọc tệp CSV với các giá trị khuyết thiếu được chuẩn hóa."""
    if na_values is None:
        na_values = ['?', 'NA', 'null', '']
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu: {file_path}")
    if chunksize:
        return pd.read_csv(file_path, na_values=na_values, chunksize=chunksize, low_memory=False)
    return pd.read_csv(file_path, na_values=na_values, low_memory=False)

def load_all_oulad(raw_dir: str = 'data/raw') -> Dict[str, pd.DataFrame]:
    """Tải toàn bộ các bảng nhỏ và vừa của OULAD vào bộ nhớ."""
    tables = {}
    csv_files = ['courses.csv', 'assessments.csv', 'vle.csv', 
                 'studentInfo.csv', 'studentRegistration.csv', 'studentAssessment.csv']
    for f in csv_files:
        name = f.replace('.csv', '')
        path = os.path.join(raw_dir, f)
        print(f"[DataLoader] Đang tải bảng {name} từ {path}...")
        tables[name] = load_table(path)
    return tables

def stream_student_vle(raw_dir: str = 'data/raw', chunksize: int = 1000000) -> Generator[pd.DataFrame, None, None]:
    """Đọc luồng dữ liệu studentVle.csv theo từng chunk để tối ưu bộ nhớ RAM."""
    path = os.path.join(raw_dir, 'studentVle.csv')
    print(f"[DataLoader] Khởi tạo luồng đọc studentVle (chunksize={chunksize:,})...")
    return load_table(path, chunksize=chunksize)

if __name__ == '__main__':
    tables = load_all_oulad('data/raw')
    for k, v in tables.items():
        print(f"{k}: {v.shape}")
