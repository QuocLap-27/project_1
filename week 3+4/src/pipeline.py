# -*- coding: utf-8 -*-
"""
Module: pipeline.py
Dự án: SMART LMS & GENERATIVE AI
Chức năng: Pipeline đầu-cuối thực thi khảo sát, làm sạch, trích xuất đặc trưng và xuất dữ liệu.
"""

import argparse
import os
import sys
from data_loader import load_all_oulad, stream_student_vle
from data_inspection import audit_dataset_quality, check_keys_integrity
from feature_engineering import extract_features_for_presentation
from data_cleaning import clean_student_features
from eda import generate_eda_reports_and_plots
from export_data import save_processed_datasets

def run_pipeline(module: str = 'FFF', presentation: str = '2013J', raw_dir: str = 'data/raw'):
    print(f"=== BẮT ĐẦU PIPELINE TIỀN XỬ LÝ DỮ LIỆU ({module} - {presentation}) ===")
    
    # Bước 1: Tải dữ liệu
    tables = load_all_oulad(raw_dir)
    
    # Bước 2: Kiểm tra dữ liệu
    print("[Pipeline] Đang kiểm tra chất lượng dữ liệu...")
    audit = audit_dataset_quality(tables)
    keys = check_keys_integrity(tables)
    print("[Pipeline] Kiểm tra khóa thành công:", keys)
    
    # Bước 3: Trích xuất đặc trưng
    print("[Pipeline] Đang trích xuất đặc trưng...")
    features = extract_features_for_presentation(
        tables=tables,
        code_module=module,
        code_presentation=presentation,
        vle_stream_func=stream_student_vle,
        raw_dir=raw_dir
    )
    
    # Bước 4: Làm sạch và chuẩn hóa
    print("[Pipeline] Đang làm sạch và chuẩn hóa dữ liệu...")
    cleaned = clean_student_features(features)
    
    # Bước 5: Phân tích tương quan
    print("[Pipeline] Đang phân tích tương quan EDA...")
    corr = generate_eda_reports_and_plots(cleaned, output_fig_dir='reports/figures')
    
    # Bước 6: Xuất dữ liệu
    print("[Pipeline] Đang xuất dữ liệu ra file CSV...")
    save_processed_datasets(cleaned, base_dir='.')
    
    print(f"=== HOÀN THÀNH PIPELINE XỬ LÝ DỮ LIỆU! Bộ dữ liệu gồm {len(cleaned)} sinh viên, {cleaned.shape[1]} thuộc tính ===")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chạy pipeline tiền xử lý dữ liệu OULAD Smart LMS')
    parser.add_argument('--module', type=str, default='FFF', help='Mã môn học (mặc định FFF)')
    parser.add_argument('--presentation', type=str, default='2013J', help='Mã kỳ học (mặc định 2013J)')
    parser.add_argument('--raw_dir', type=str, default='data/raw', help='Thư mục chứa tệp CSV OULAD')
    args = parser.parse_args()
    
    run_pipeline(module=args.module, presentation=args.presentation, raw_dir=args.raw_dir)
