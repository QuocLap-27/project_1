# -*- coding: utf-8 -*-
"""
Module: data_inspection.py
Dự án: SMART LMS & GENERATIVE AI
Chức năng: Kiểm tra tính toàn vẹn, missing values, duplicates, kiểu dữ liệu và khóa liên kết.
"""

import os
import pandas as pd
from typing import Dict, Any

def audit_dataset_quality(tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Kiểm tra toàn diện chất lượng dữ liệu của các bảng OULAD."""
    report = {}
    for name, df in tables.items():
        row_count = len(df)
        col_count = len(df.columns)
        null_counts = df.isnull().sum()
        missing_dict = null_counts[null_counts > 0].to_dict()
        dup_count = df.duplicated().sum()
        
        report[name] = {
            'rows': row_count,
            'cols': col_count,
            'missing_values': missing_dict,
            'duplicates': dup_count,
            'dtypes': {c: str(t) for c, t in df.dtypes.items()}
        }
    return report

def check_keys_integrity(tables: Dict[str, pd.DataFrame]) -> Dict[str, bool]:
    """Kiểm tra tính duy nhất của khóa chính và tính toàn vẹn tham chiếu khóa ngoại."""
    results = {}
    si = tables.get('studentInfo')
    sr = tables.get('studentRegistration')
    assess = tables.get('assessments')
    sa = tables.get('studentAssessment')
    vle = tables.get('vle')
    
    if si is not None and sr is not None:
        si_keys = set(zip(si['code_module'], si['code_presentation'], si['id_student']))
        sr_keys = set(zip(sr['code_module'], sr['code_presentation'], sr['id_student']))
        results['si_sr_key_match'] = (si_keys == sr_keys)
        results['si_unique_pk'] = (len(si) == len(si_keys))
        
    if assess is not None and sa is not None:
        assess_ids = set(assess['id_assessment'])
        sa_assess_ids = set(sa['id_assessment'])
        results['assess_pk_unique'] = assess['id_assessment'].is_unique
        results['sa_fk_valid'] = sa_assess_ids.issubset(assess_ids)
        
    if vle is not None:
        results['vle_pk_unique'] = vle['id_site'].is_unique
        
    return results

if __name__ == '__main__':
    from data_loader import load_all_oulad
    tables = load_all_oulad('data/raw')
    report = audit_dataset_quality(tables)
    keys_report = check_keys_integrity(tables)
    print("=== BÁO CÁO CHẤT LƯỢNG DỮ LIỆU ===")
    for k, v in report.items():
        print(f"{k}: {v['rows']:,} rows, {v['duplicates']} duplicates, missing: {v['missing_values']}")
    print("=== TÍNH TOÀN VẸN KHÓA ===", keys_report)
