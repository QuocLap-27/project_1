# TRIỂN KHAI TUẦN 5 VÀ TUẦN 6 – SMART LMS & GENERATIVE AI
**Dự án:** Phân Tích Dữ Liệu Người Học và Tạo Nội Dung Học Tập Tự Động (Smart LMS & Generative AI)  
**Sinh viên thực hiện:** Bế Quốc Lập (MSSV: 202416542)  
**Môn học trọng tâm:** OULAD `FFF` – Học kỳ `2013J` (2.283 sinh viên) | **Môn đối chứng:** `BBB` – `2013J`

## 1. Cấu trúc thư mục `Week5_Week6_Deliverables/`
- `A_source_code/`:
  - `01_data_validation.ipynb`: Khảo sát 37 cột `student_features.csv`, xác minh 24 đặc trưng định lượng và kiểm toán Data Leakage.
  - `02_model_training.ipynb`: Huấn luyện Logistic Regression, Random Forest, Decision Tree với Stratified Split (80/20, `random_state=42`) & Stratified 5-Fold CV.
  - `03_model_evaluation.ipynb`: Đánh giá 8 nhóm chỉ số, tối ưu Decision Threshold (`0.35`) và phân tích Feature Importance & Permutation Importance.
  - `train_models.py`: Script huấn luyện và xuất mô hình tự động.
  - `predict.py`: Module suy luận (`SmartLMSRiskPredictor`) kiểm tra 24 đặc trưng đầu vào, trả về xác suất rủi ro và mức cảnh báo `LOW` / `MEDIUM` / `HIGH`.
- `B_models_and_results/`:
  - `best_model.joblib` & `best_model_early_d100.joblib`: Pipeline đóng gói (`SimpleImputer` + `StandardScaler` + `RandomForestClassifier`).
  - `model_metadata.json`: Siêu dữ liệu mô hình, thứ tự 24 đặc trưng, siêu tham số tối ưu và ngưỡng quyết định (`0.35`).
  - `model_comparison.csv`, `threshold_comparison.csv`, `classification_report.txt`, `feature_importance_w4.csv`, `feature_importance_d100.csv`, `student_features_early_d100.csv`.
- `C_figures/`:
  - `confusion_matrix.png`, `roc_curve.png`, `precision_recall_curve.png`, `feature_importance.png`, `model_comparison.png`, `threshold_analysis.png`.
- `D_curriculum_mapping/`:
  - `curriculum_topics.csv` (9 bản ghi), `assessment_topic_mapping.csv` (25 bản ghi), `vle_topic_mapping.csv` (529 bản ghi của `FFF 2013J`), `curriculum_mapping_schema.sql`, `curriculum_mapping_erd.png`.
- `E_reports/`:
  - `Bao_Cao_Tuan_5_6_Smart_LMS.docx`: Báo cáo học thuật đầy đủ 9 mục kèm bảng số liệu và hình ảnh thực nghiệm.
