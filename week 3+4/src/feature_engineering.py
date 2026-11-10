# -*- coding: utf-8 -*-
"""
Module: feature_engineering.py
Dự án: SMART LMS & GENERATIVE AI
Chức năng: Trích xuất 25 đặc trưng học tập (Điểm số, Hành vi VLE, Nhân khẩu học) và Biến mục tiêu.
Đảm bảo chống rò rỉ dữ liệu (Data Leakage).
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

def extract_features_for_presentation(
    tables: Dict[str, pd.DataFrame],
    code_module: str = 'FFF',
    code_presentation: str = '2013J',
    vle_stream_func = None,
    raw_dir: str = 'data/raw'
) -> pd.DataFrame:
    """Trích xuất toàn bộ bộ đặc trưng cho một môn học và học kỳ cụ thể."""
    si = tables['studentInfo']
    sr = tables['studentRegistration']
    assess = tables['assessments']
    sa = tables['studentAssessment']
    vle = tables['vle']

    # 1. Lọc theo môn học và kỳ học
    si_sub = si[(si['code_module'] == code_module) & (si['code_presentation'] == code_presentation)].copy()
    sr_sub = sr[(sr['code_module'] == code_module) & (sr['code_presentation'] == code_presentation)].copy()
    assess_sub = assess[(assess['code_module'] == code_module) & (assess['code_presentation'] == code_presentation)].copy()

    # LOẠI BỎ ĐIỂM THI CUỐI KỲ (EXAM) ĐỂ TRÁNH RÒ RỈ DỮ LIỆU (DATA LEAKAGE)
    in_course_assess = assess_sub[assess_sub['assessment_type'] != 'Exam']
    total_assess_num = len(in_course_assess)

    # 2. Xử lý nhóm đặc trưng Điểm số (Group A)
    sa_sub = sa[sa['id_assessment'].isin(in_course_assess['id_assessment'])].copy()
    sa_sub['score'] = pd.to_numeric(sa_sub['score'], errors='coerce')
    sa_sub = sa_sub.merge(in_course_assess[['id_assessment', 'assessment_type', 'date', 'weight']], on='id_assessment', how='left')
    sa_sub['is_late'] = (sa_sub['date_submitted'] > sa_sub['date']).astype(int)
    sa_sub['delay'] = sa_sub['date_submitted'] - sa_sub['date']

    sa_grouped = sa_sub.groupby('id_student').agg(
        num_assessments=('id_assessment', 'count'),
        avg_score=('score', 'mean'),
        min_score=('score', 'min'),
        max_score=('score', 'max'),
        score_std=('score', 'std'),
        late_submission_count=('is_late', 'sum'),
        avg_submission_delay=('delay', 'mean'),
        weighted_score=('score', lambda s: (s * sa_sub.loc[s.index, 'weight'] / 100.0).sum())
    ).reset_index()
    sa_grouped['score_std'] = sa_grouped['score_std'].fillna(0.0)

    tma_sa = sa_sub[sa_sub['assessment_type'] == 'TMA'].groupby('id_student')['score'].mean().rename('tma_avg_score')
    cma_sa = sa_sub[sa_sub['assessment_type'] == 'CMA'].groupby('id_student')['score'].mean().rename('cma_avg_score')
    sa_grouped = sa_grouped.merge(tma_sa, on='id_student', how='left').merge(cma_sa, on='id_student', how='left')

    # 3. Xử lý nhóm đặc trưng Hành vi VLE (Group B)
    vle_sub = vle[(vle['code_module'] == code_module) & (vle['code_presentation'] == code_presentation)][['id_site', 'activity_type']]
    vle_dict = dict(zip(vle_sub['id_site'], vle_sub['activity_type']))

    vle_rows = []
    chunk_iter = vle_stream_func(raw_dir=raw_dir) if vle_stream_func else pd.read_csv(f'{raw_dir}/studentVle.csv', na_values=['?'], chunksize=1000000)
    for chunk in chunk_iter:
        c_sub = chunk[(chunk['code_module'] == code_module) & (chunk['code_presentation'] == code_presentation)].copy()
        if len(c_sub) > 0:
            c_sub['activity_type'] = c_sub['id_site'].map(vle_dict)
            vle_rows.append(c_sub)
    df_vle_sub = pd.concat(vle_rows, ignore_index=True)

    vle_agg = df_vle_sub.groupby('id_student').agg(
        total_clicks=('sum_click', 'sum'),
        active_days=('date', 'nunique'),
        unique_learning_resources=('id_site', 'nunique'),
        early_clicks_pre_course=('sum_click', lambda c: c[df_vle_sub.loc[c.index, 'date'] < 0].sum()),
        clicks_first_half=('sum_click', lambda c: c[(df_vle_sub.loc[c.index, 'date'] >= 0) & (df_vle_sub.loc[c.index, 'date'] <= 100)].sum()),
        clicks_oucontent=('sum_click', lambda c: c[df_vle_sub.loc[c.index, 'activity_type'] == 'oucontent'].sum()),
        clicks_forumng=('sum_click', lambda c: c[df_vle_sub.loc[c.index, 'activity_type'] == 'forumng'].sum()),
        clicks_quiz=('sum_click', lambda c: c[df_vle_sub.loc[c.index, 'activity_type'].isin(['quiz', 'externalquiz'])].sum()),
        clicks_resource_url=('sum_click', lambda c: c[df_vle_sub.loc[c.index, 'activity_type'].isin(['resource', 'url', 'subpage'])].sum())
    ).reset_index()

    vle_agg['average_clicks_per_day'] = vle_agg['total_clicks'] / vle_agg['active_days']

    # 4. Hợp nhất dữ liệu và xử lý nhóm đặc trưng Học tập & Nhân khẩu (Group C)
    features = si_sub.merge(sr_sub[['id_student', 'date_registration', 'date_unregistration']], on='id_student', how='left')
    features = features.merge(sa_grouped, on='id_student', how='left')
    features = features.merge(vle_agg, on='id_student', how='left')

    # Xử lý giá trị mặc định cho sinh viên chưa có hoạt động
    features['assessment_completion_rate'] = (features['num_assessments'].fillna(0) / total_assess_num).clip(0, 1)
    features['num_assessments'] = features['num_assessments'].fillna(0).astype(int)
    for col in ['avg_score', 'min_score', 'max_score', 'score_std', 'weighted_score', 'tma_avg_score', 'cma_avg_score', 'avg_submission_delay']:
        features[col] = features[col].fillna(0.0)
    features['late_submission_count'] = features['late_submission_count'].fillna(0).astype(int)

    for col in ['total_clicks', 'active_days', 'unique_learning_resources', 'early_clicks_pre_course', 'clicks_first_half', 'clicks_oucontent', 'clicks_forumng', 'clicks_quiz', 'clicks_resource_url']:
        features[col] = features[col].fillna(0).astype(int)
    features['average_clicks_per_day'] = features['average_clicks_per_day'].fillna(0.0)

    features['has_unregistered'] = features['date_unregistration'].notnull().astype(int)
    median_reg = features['date_registration'].median()
    features['date_registration'] = features['date_registration'].fillna(median_reg)

    # 5. XÁC ĐỊNH BIẾN MỤC TIÊU (TARGET VARIABLE)
    features['is_at_risk'] = features['final_result'].isin(['Fail', 'Withdrawn']).astype(int)

    return features
