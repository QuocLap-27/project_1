# -*- coding: utf-8 -*-
"""
Module: eda.py
Dự án: SMART LMS & GENERATIVE AI
Chức năng: Phân tích khám phá dữ liệu (EDA), tính ma trận tương quan và sinh biểu đồ trực quan hóa.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_eda_reports_and_plots(df: pd.DataFrame, output_fig_dir: str = 'reports/figures') -> pd.DataFrame:
    """Sinh toàn bộ 6 biểu đồ EDA và bảng tương quan Pearson/Spearman."""
    os.makedirs(output_fig_dir, exist_ok=True)
    
    # 1. Tính tương quan
    num_cols = [
        'avg_score', 'weighted_score', 'assessment_completion_rate', 'num_assessments', 
        'tma_avg_score', 'cma_avg_score', 'min_score', 'max_score', 'score_std',
        'total_clicks', 'active_days', 'average_clicks_per_day', 'unique_learning_resources',
        'early_clicks_pre_course', 'clicks_first_half', 'clicks_oucontent', 'clicks_forumng', 
        'clicks_quiz', 'clicks_resource_url', 'num_of_prev_attempts', 'studied_credits',
        'date_registration', 'late_submission_count', 'avg_submission_delay'
    ]
    
    pearson = df[num_cols].apply(lambda s: s.corr(df['is_at_risk'], method='pearson'))
    spearman = df[num_cols].apply(lambda s: s.corr(df['is_at_risk'], method='spearman'))
    
    corr_df = pd.DataFrame({
        'feature': num_cols,
        'pearson_corr': pearson,
        'spearman_corr': spearman
    }).sort_values(by='spearman_corr', ascending=True)
    
    return corr_df
