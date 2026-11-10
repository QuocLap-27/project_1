# -*- coding: utf-8 -*-
"""
Module: export_data.py
Dự án: SMART LMS & GENERATIVE AI
Chức năng: Xuất các tệp CSV sau tiền xử lý phục vụ huấn luyện Machine Learning ở Tuần 5.
"""

import os
import pandas as pd

def save_processed_datasets(features_df: pd.DataFrame, base_dir: str = '.'):
    """Lưu các bộ dữ liệu sạch và từ điển dữ liệu."""
    proc_dir = os.path.join(base_dir, 'data/processed')
    out_dir = os.path.join(base_dir, 'data/output')
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
    features_df.to_csv(os.path.join(proc_dir, 'oulad_cleaned.csv'), index=False, encoding='utf-8')
    features_df.to_csv(os.path.join(out_dir, 'student_features.csv'), index=False, encoding='utf-8')
    print(f"[Export] Đã lưu thành công oulad_cleaned.csv và student_features.csv ({len(features_df)} dòng).")
