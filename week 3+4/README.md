# SMART LMS & GENERATIVE AI
## Phân Tích Dữ Liệu Người Học Và Tạo Nội Dung Tự Động
**Sinh viên thực hiện:** Bế Quốc Lập (MSV: 202416542)  
**Giai đoạn:** Tuần 3 (Thu thập & Định nghĩa dữ liệu) và Tuần 4 (Tiền xử lý dữ liệu & EDA)

---

## 1. Cấu trúc dự án


---

## 2. Hướng dẫn cài đặt và thực thi trên Windows

### Bước 1: Chuẩn bị môi trường Python


### Bước 2: Chạy Pipeline xử lý dữ liệu đầu-cuối


### Bước 3: Đưa dữ liệu vào Microsoft SQL Server
1. Mở **SQL Server Management Studio (SSMS)**.
2. Mở tệp  và nhấn  để khởi tạo CSDL  cùng toàn bộ bảng.
3. Thực hiện lệnh  hoặc chạy tiện ích  theo hướng dẫn ở cuối file SQL để nạp dữ liệu.

---

## 3. Chuyển tiếp sang Tuần 5 (Huấn luyện Machine Learning)
Bộ dữ liệu  đã sẵn sàng 100%:
- 2,283 sinh viên môn FFF kỳ 2013J.
- 24 đặc trưng đầu vào định lượng (Nhóm Điểm số, Hành vi VLE, Nhân khẩu học).
- Biến mục tiêu  (52% At-Risk vs 48% Non-Risk) chuẩn hóa nhị phân.
- Không rò rỉ dữ liệu (không chứa điểm thi Exam, không chứa final_result, không chứa id_student).
- Đủ điều kiện để huấn luyện và so sánh trực tiếp 3 thuật toán: **Random Forest, Logistic Regression, Decision Tree/XGBoost** ở Tuần 5.
