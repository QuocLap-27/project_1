# SMART LMS & GENERATIVE AI – TUẦN 7 & TUẦN 8

**Sinh viên thực hiện:** Bế Quốc Lập – **MSSV:** `202416542`  
**Môn học thực nghiệm:** `IT1108 – Nhập môn lập trình` (Ngôn ngữ C), ánh xạ từ dữ liệu `OULAD FFF – 2013J`  
**Nội dung:** Kiểm thử Generative AI (Gemini API, 3 kịch bản Prompt) và kết nối pipeline **ML Prediction → Topic Mapping → GenAI** kèm giao diện Web UI demo.

---

## 1. Cấu trúc thư mục

```text
week 7+8/
├── .env.example                            # Mẫu biến môi trường (copy thành .env và điền API Key)
├── requirements.txt                        # Thư viện Python cần thiết
├── README.md
├── BAO_CAO_TUAN_7_8_SMART_LMS.md           # Báo cáo tổng kết Tuần 7–8
├── src/
│   ├── config.py                           # Đọc .env, cấu hình model, timeout, retry, che API Key
│   ├── curriculum_loader.py                # Đọc read-only bảng Mapping từ week 5+6/
│   ├── schemas.py                          # responseSchema rút gọn & kiểm tra JSON chuẩn 6 trường
│   ├── pii_guard.py                        # Kiểm tra và loại bỏ thông tin định danh (PII)
│   ├── prompt_builder.py                   # Kế thừa build_prompt_scenarios() tạo 3 kịch bản Prompt
│   ├── mock_provider.py                    # Ngân hàng câu hỏi C IT1108 & mô phỏng lỗi khi test Mock
│   ├── gemini_client.py                    # Gọi Gemini API / Mock, xử lý retry 503, Timeout, JSON
│   ├── ml_predictor_bridge.py              # Kết nối predict.py & best_model.joblib (24 đặc trưng)
│   ├── topic_mapper.py                     # Ánh xạ 6 khối IT1108 & tính phân vị từ OULAD FFF-2013J
│   ├── ml_genai_pipeline.py                # Pipeline end-to-end ML -> Mapping -> GenAI
│   └── experiment_runner.py                # Chạy thực nghiệm so sánh 3 kịch bản Prompt
├── evaluation/
│   ├── it1108_benchmark_dataset.json       # Bộ dữ liệu kiểm thử chủ đề IT1108
│   ├── compare_prompt_scenarios.py         # So sánh chất lượng đầu ra 3 kịch bản Prompt
│   ├── verify_ml_adapter.py                # Kiểm chứng tương thích best_model.joblib trên 2.283 mẫu
│   └── run_week8_acceptance_demos.py       # Chạy 3 kịch bản demo qua Web UI API
├── playground/
│   ├── app.py                              # Flask Web UI demo Tuần 7 & Tuần 8 (cổng 5057)
│   └── templates/index.html                # Giao diện trực quan trên trình duyệt
├── tests/
│   ├── sample_profiles.json                # Hồ sơ học tập mô phỏng
│   ├── test_schema_and_nfr.py              # Unit test Tuần 7 (20 test)
│   ├── test_week8_pipeline.py              # Unit test pipeline Tuần 8 (8 test)
│   ├── test_webui_and_acceptance.py        # Unit test Web UI & xử lý lỗi (8 test)
│   └── test_live_gemini.py                 # Integration test gọi Live Gemini API
└── outputs/                                # Kết quả thực nghiệm & demo (JSON / CSV)
```

> Lưu ý: Không chỉnh sửa dữ liệu hay mã nguồn của `w1/`, `w2/`, `week 3+4/`, `week 5+6/` và giữ nguyên cấu trúc 3 bảng Mapping (`curriculum_topics`, `assessment_topic_mapping`, `vle_topic_mapping`).

---

## 2. Cài đặt môi trường

Cài các thư viện cho thư mục `week 7+8/`:

```bash
pip install -r "week 7+8/requirements.txt"
```

*(Lưu ý tương thích ML: `best_model.joblib` được huấn luyện ở Tuần 5–6 bằng Python `3.11.2` và `scikit-learn 1.5.2`. Nếu môi trường chạy dùng phiên bản `scikit-learn` mới hơn, `MLPredictorBridge` tự động dùng adapter đọc đúng trọng số đã huấn luyện để suy luận; khi cần tái lập môi trường huấn luyện gốc nên tạo `venv` với Python `3.11` và `scikit-learn==1.5.2`.)*

---

## 3. Cấu hình Gemini API Key

1. Sao chép `week 7+8/.env.example` thành `week 7+8/.env`.
2. Điền API Key Google AI Studio vào biến `GEMINI_API_KEY` (không commit file `.env` lên GitHub).
3. Các cấu hình mặc định trong `.env`:
   - `GEMINI_MODEL=gemini-3.5-flash` và `GEMINI_FALLBACK_MODEL=gemini-3.5-flash-lite`
   - `GEMINI_USE_RESPONSE_SCHEMA=true`
   - `GEMINI_TIMEOUT_SECONDS=20`, `GEMINI_MAX_RETRIES=2`, `GEMINI_MAX_JSON_RETRIES=1`

---

## 4. Hướng dẫn chạy nhanh

### Chạy Web UI Demo (cổng `5057`)
```bash
python "week 7+8/playground/app.py"
```
Truy cập `http://127.0.0.1:5057` để thử trực tiếp 3 kịch bản demo Tuần 8 hoặc chuyển sang tab Playground Tuần 7.

### Chạy kiểm thử và demo qua dòng lệnh
```bash
# Chạy toàn bộ Unit Test offline (36 test)
python -m unittest discover -s "week 7+8/tests" -p "test_schema_and_nfr.py" -v
python -m unittest discover -s "week 7+8/tests" -p "test_week8_pipeline.py" -v
python -m unittest discover -s "week 7+8/tests" -p "test_webui_and_acceptance.py" -v

# Kiểm chứng độ tương đương tiền xử lý & hoạt động ML adapter
python "week 7+8/evaluation/verify_ml_adapter.py"

# Chạy 3 kịch bản demo nghiệm thu (chọn --mode mock hoặc --mode live)
python "week 7+8/evaluation/run_week8_acceptance_demos.py" --mode mock
python "week 7+8/evaluation/run_week8_acceptance_demos.py" --mode live --timeout 45
```
