# -*- coding: utf-8 -*-
"""
Module: data_cleaning.py
Dự án: SMART LMS & GENERATIVE AI
Chức năng: Làm sạch, điền khuyết thiếu, xử lý ngoại lai và chuẩn hóa kiểu dữ liệu.
"""

import pandas as pd
import numpy as np

def clean_student_features(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch bảng đặc trưng sinh viên."""
    cleaned_df = df.copy()

    # 1. Điền missing imd_band
    cleaned_df['imd_band'] = cleaned_df['imd_band'].fillna('Missing')

    # 2. Chuẩn hóa kiểu dữ liệu
    int_cols = ['id_student', 'num_of_prev_attempts', 'studied_credits', 
                'num_assessments', 'late_submission_count', 'total_clicks', 
                'active_days', 'unique_learning_resources', 'early_clicks_pre_course',
                'clicks_first_half', 'clicks_oucontent', 'clicks_forumng', 
                'clicks_quiz', 'clicks_resource_url', 'has_unregistered', 'is_at_risk']
    for c in int_cols:
        if c in cleaned_df.columns:
            cleaned_df[c] = cleaned_df[c].astype(int)

    float_cols = ['date_registration', 'avg_score', 'weighted_score', 'min_score', 
                  'max_score', 'score_std', 'tma_avg_score', 'cma_avg_score', 
                  'avg_submission_delay', 'assessment_completion_rate', 'average_clicks_per_day']
    for c in float_cols:
        if c in cleaned_df.columns:
            cleaned_df[c] = cleaned_df[c].astype(float)

    return cleaned_df
