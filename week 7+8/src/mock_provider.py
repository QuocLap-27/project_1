# -*- coding: utf-8 -*-
"""
Module: mock_provider.py
Dự án: SMART LMS & GENERATIVE AI (Tuần 7+8 - Kiểm thử GenAI)
Chức năng:
- Cung cấp ngân hàng câu hỏi C chuẩn học phần IT1108 (đủ 6 khối IT1108_KB01..IT1108_KB06, mỗi khối >= 5 câu)
  để kiểm thử toàn bộ luồng Prompt -> JSON Validator -> UI khi chưa có API key.
- Xuất JSON đúng 6 trường chuẩn Prompt 2: question, options, correct_answer, explanation, topic, difficulty.
- Mô phỏng các tình huống lỗi (timeout, rate_limit, malformed_json, schema_error) phục vụ test NFR-01 & NFR-03.
"""

from __future__ import annotations

import json
import socket
from typing import Any, Dict, List, Optional, Tuple


class MockRateLimitError(RuntimeError):
    """Mô phỏng lỗi HTTP 429 RESOURCE_EXHAUSTED từ Gemini API."""


class MockServiceUnavailableError(RuntimeError):
    """Mô phỏng lỗi HTTP 503 UNAVAILABLE từ Gemini API."""


# Ngân hàng câu hỏi C chuẩn cho 6 khối kiến thức IT1108 (mỗi khối có >= 5 câu độc lập).
_QUESTION_BANK: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
    "IT1108_KB01": {
        "guided_basic": [
            {
                "topic": "Khối 1: Giải quyết bài toán, Thuật toán & Tổng quan lập trình",
                "difficulty": "easy",
                "question_type": "guided_basic",
                "question": "Trong quy trình giải quyết bài toán bằng máy tính, thứ tự các bước cơ bản nào sau đây là đúng?",
                "code_snippet": "",
                "options": {
                    "A": "Xác định bài toán -> Thiết kế thuật toán -> Viết chương trình -> Kiểm thử & sửa lỗi",
                    "B": "Viết chương trình -> Xác định bài toán -> Kiểm thử -> Thiết kế thuật toán",
                    "C": "Kiểm thử -> Thiết kế thuật toán -> Xác định bài toán -> Viết chương trình",
                    "D": "Thiết kế giao diện -> Viết chương trình -> Bỏ qua bước kiểm thử",
                },
                "correct_answer": "A",
                "hint": "Gợi ý từng bước: Bước 1 cần xác định rõ Input/Output, Bước 2 thiết kế thuật toán, sau đó mới lập trình C và kiểm thử.",
                "explanation": "Quy trình chuẩn gồm 4 giai đoạn nối tiếp: (1) Xác định Input/Output của bài toán, (2) Thiết kế thuật toán bằng mã giả hoặc sơ đồ khối, (3) Cài đặt thành chương trình C, (4) Biên dịch và kiểm thử với nhiều bộ dữ liệu.",
            },
            {
                "topic": "Khối 1: Giải quyết bài toán, Thuật toán & Tổng quan lập trình",
                "difficulty": "easy",
                "question_type": "guided_basic",
                "question": "Trong sơ đồ khối (Flowchart), ký hiệu hình thoi được sử dụng để biểu diễn thao tác nào?",
                "code_snippet": "",
                "options": {
                    "A": "Bắt đầu hoặc kết thúc thuật toán",
                    "B": "Nhập hoặc xuất dữ liệu",
                    "C": "Kiểm tra điều kiện rẽ nhánh (Đúng/Sai)",
                    "D": "Thực hiện phép gán hoặc tính toán tuần tự",
                },
                "correct_answer": "C",
                "hint": "Gợi ý từng bước: Hình oval dùng cho Bắt đầu/Kết thúc; hình chữ nhật cho tính toán; hình thoi có 2 nhánh Đúng/Sai.",
                "explanation": "Trong chuẩn biểu diễn sơ đồ khối IT1108: hình oval là Bắt đầu/Kết thúc, hình bình hành là Nhập/Xuất, hình chữ nhật là Xử lý/Tính toán, còn hình thoi biểu diễn nút kiểm tra điều kiện rẽ nhánh.",
            },
        ],
        "bug_trace": [
            {
                "topic": "Khối 1: Giải quyết bài toán, Thuật toán & Tổng quan lập trình",
                "difficulty": "medium",
                "question_type": "code_trace",
                "question": "Cho đoạn mã giả thuật toán dưới đây. Khi nhập n = 4, giá trị cuối cùng của s là bao nhiêu?",
                "code_snippet": "s <- 0\ni <- 1\nTrong khi i <= n:\n    s <- s + i\n    i <- i + 1\nXuat s",
                "options": {"A": "6", "B": "10", "C": "4", "D": "15"},
                "correct_answer": "B",
                "hint": "Hãy lập bảng lần vết giá trị (i, s) qua từng bước lặp từ i = 1 đến i = 4.",
                "explanation": "Lần vết từng bước với n = 4: Khởi tạo s=0, i=1. Vòng 1: s=1, i=2. Vòng 2: s=3, i=3. Vòng 3: s=6, i=4. Vòng 4: s=10, i=5. Khi i=5 > 4 vòng lặp dừng và xuất s = 10.",
            },
            {
                "topic": "Khối 1: Giải quyết bài toán, Thuật toán & Tổng quan lập trình",
                "difficulty": "medium",
                "question_type": "bug_analysis",
                "question": "Đoạn mã giả tìm số lớn nhất trong 2 số a, b dưới đây gặp lỗi logic gì nếu a = 5 và b = 5?",
                "code_snippet": "Neu a > b:\n    max <- a\nNeu b > a:\n    max <- b\nXuat max",
                "options": {
                    "A": "Khi a == b, không nhánh nào gán giá trị cho biến max",
                    "B": "Lệnh Neu không thể so sánh hai số nguyên",
                    "C": "Biến max luôn nhận giá trị a trong mọi trường hợp",
                    "D": "Thuật toán rơi vào vòng lặp vô hạn",
                },
                "correct_answer": "A",
                "hint": "Thử thế a = 5, b = 5 vào cả hai điều kiện a > b và b > a.",
                "explanation": "Khi a == b, cả (a > b) và (b > a) đều sai nên biến max chưa được gán giá trị (dữ liệu rác). Cần dùng cấu trúc Neu a >= b ... Nguoc lai ... để bao phủ đủ mọi trường hợp.",
            },
        ],
        "standard": [
            {
                "topic": "Khối 1: Giải quyết bài toán, Thuật toán & Tổng quan lập trình",
                "difficulty": "medium",
                "question_type": "concept",
                "question": "Tính chất 'Tính dừng' (Finiteness) của một thuật toán có ý nghĩa gì?",
                "code_snippet": "",
                "options": {
                    "A": "Thuật toán phải chạy trong thời gian dưới 1 giây",
                    "B": "Thuật toán phải kết thúc sau một số hữu hạn bước với mọi đầu vào hợp lệ",
                    "C": "Thuật toán không được chứa bất kỳ vòng lặp nào",
                    "D": "Thuật toán chỉ dùng cho bài toán số học",
                },
                "correct_answer": "B",
                "hint": "Nhớ lại 5 tính chất cơ bản của thuật toán: đầu vào, đầu ra, tính xác định, tính dừng và tính hiệu quả.",
                "explanation": "Tính dừng đảm bảo thuật toán không bị lặp vô tận mà luôn kết thúc sau số bước hữu hạn để cho ra kết quả đúng.",
            },
        ],
    },
    "IT1108_KB02": {
        "guided_basic": [
            {
                "topic": "Khối 2: Ngôn ngữ C cơ bản, Hệ đếm, Kiểu dữ liệu, Vào/ra & Biểu thức",
                "difficulty": "easy",
                "question_type": "guided_basic",
                "question": "Trong ngôn ngữ C, hàm nào sau đây dùng để đọc một số nguyên từ bàn phím vào biến x?",
                "code_snippet": "int x;\n// Cần đọc giá trị nguyên vào biến x",
                "options": {
                    "A": "scanf(\"%d\", &x);",
                    "B": "scanf(\"%d\", x);",
                    "C": "printf(\"%d\", &x);",
                    "D": "input(\"%d\", &x);",
                },
                "correct_answer": "A",
                "hint": "Gợi ý từng bước: Dùng scanf, định dạng %d cho kiểu int và đặt toán tử địa chỉ & trước biến x.",
                "explanation": "Hàm scanf cần địa chỉ vùng nhớ của biến nên phải viết &x. Thiếu dấu & sẽ truyền giá trị rác làm địa chỉ và gây lỗi khi chạy chương trình.",
            },
            {
                "topic": "Khối 2: Ngôn ngữ C cơ bản, Hệ đếm, Kiểu dữ liệu, Vào/ra & Biểu thức",
                "difficulty": "easy",
                "question_type": "guided_basic",
                "question": "Số nhị phân 1010 (cơ số 2) tương ứng với giá trị nào trong hệ thập phân (cơ số 10)?",
                "code_snippet": "",
                "options": {"A": "8", "B": "10", "C": "12", "D": "14"},
                "correct_answer": "B",
                "hint": "Gợi ý từng bước: Nhân từng bit từ phải sang trái với lũy thừa của 2: 1*2^3 + 0*2^2 + 1*2^1 + 0*2^0.",
                "explanation": "Áp dụng công thức quy đổi nhị phân sang thập phân: 1*8 + 0*4 + 1*2 + 0*1 = 8 + 2 = 10.",
            },
        ],
        "bug_trace": [
            {
                "topic": "Khối 2: Ngôn ngữ C cơ bản, Hệ đếm, Kiểu dữ liệu, Vào/ra & Biểu thức",
                "difficulty": "medium",
                "question_type": "bug_analysis",
                "question": "Khi chạy đoạn chương trình C dưới đây, kết quả in ra màn hình là gì và vì sao?",
                "code_snippet": "#include <stdio.h>\nint main() {\n    int a = 5, b = 2;\n    float kq = a / b;\n    printf(\"%.1f\", kq);\n    return 0;\n}",
                "options": {"A": "2.5", "B": "2.0", "C": "3.0", "D": "Lỗi biên dịch"},
                "correct_answer": "B",
                "hint": "Kiểm tra kiểu dữ liệu của hai toán hạng a và b trước khi phép chia a / b thực hiện.",
                "explanation": "Vì a và b đều có kiểu int nên phép chia a / b là chia lấy phần nguyên (5 / 2 = 2), sau đó mới ép sang float thành 2.0. Muốn ra 2.5 phải viết (float)a / b.",
            },
            {
                "topic": "Khối 2: Ngôn ngữ C cơ bản, Hệ đếm, Kiểu dữ liệu, Vào/ra & Biểu thức",
                "difficulty": "medium",
                "question_type": "code_trace",
                "question": "Đoạn chương trình C sau in ra màn hình giá trị của biến r là bao nhiêu?",
                "code_snippet": "int x = 17, y = 5;\nint r = x % y + x / y;\nprintf(\"%d\", r);",
                "options": {"A": "5", "B": "6", "C": "2", "D": "3"},
                "correct_answer": "A",
                "hint": "Tính riêng phép chia lấy dư 17 % 5 và phép chia lấy nguyên 17 / 5 rồi cộng lại.",
                "explanation": "Lần vết biểu thức: 17 % 5 cho phần dư là 2; 17 / 5 cho phần nguyên là 3. Vì vậy r = 2 + 3 = 5.",
            },
        ],
        "standard": [
            {
                "topic": "Khối 2: Ngôn ngữ C cơ bản, Hệ đếm, Kiểu dữ liệu, Vào/ra & Biểu thức",
                "difficulty": "medium",
                "question_type": "code_trace",
                "question": "Kết quả của biến c trong đoạn mã ngôn ngữ C sau là bao nhiêu?",
                "code_snippet": "int a = 3, b = 4;\nint c = ++a + b--;",
                "options": {"A": "7", "B": "8", "C": "9", "D": "6"},
                "correct_answer": "B",
                "hint": "Phân biệt tiền tố ++a (tăng trước khi dùng) và hậu tố b-- (dùng giá trị cũ rồi mới giảm).",
                "explanation": "++a tăng a từ 3 lên 4 rồi trả về 4; b-- trả về giá trị hiện tại là 4 rồi mới giảm b xuống 3. Do đó c = 4 + 4 = 8.",
            },
        ],
    },
    "IT1108_KB03": {
        "guided_basic": [
            {
                "topic": "Khối 3: Cấu trúc điều khiển rẽ nhánh & Vòng lặp",
                "difficulty": "easy",
                "question_type": "guided_basic",
                "question": "Điểm khác biệt quan trọng nhất giữa vòng lặp while và vòng lặp do...while trong C là gì?",
                "code_snippet": "",
                "options": {
                    "A": "do...while luôn thực hiện thân vòng lặp ít nhất 1 lần trước khi kiểm tra điều kiện",
                    "B": "while luôn chạy ít nhất 1 lần ngay cả khi điều kiện sai ngay từ đầu",
                    "C": "do...while không cần dấu chấm phẩy (;) ở cuối lệnh while",
                    "D": "while không thể dùng câu lệnh break bên trong",
                },
                "correct_answer": "A",
                "hint": "Gợi ý từng bước: Quan sát vị trí đặt điều kiện while(...): đặt ở đầu là kiểm tra trước; đặt ở cuối là làm trước rồi mới kiểm tra.",
                "explanation": "Vòng lặp while kiểm tra điều kiện trước khi vào thân nên có thể chạy 0 lần. Ngược lại, do...while thực hiện khối lệnh trước rồi mới xét điều kiện nên luôn chạy tối thiểu 1 lần và kết thúc bằng dấu chấm phẩy.",
            },
            {
                "topic": "Khối 3: Cấu trúc điều khiển rẽ nhánh & Vòng lặp",
                "difficulty": "easy",
                "question_type": "guided_basic",
                "question": "Để in ra các số nguyên từ 1 đến 5 bằng vòng lặp for, dòng khai báo nào sau đây viết đúng?",
                "code_snippet": "for (/* điền điều kiện */) {\n    printf(\"%d \", i);\n}",
                "options": {
                    "A": "int i = 1; i <= 5; i++",
                    "B": "int i = 1, i <= 5, i++",
                    "C": "int i = 1; i < 5; i++",
                    "D": "int i = 0; i <= 5; i++",
                },
                "correct_answer": "A",
                "hint": "Gợi ý từng bước: 3 thành phần trong for(...) phân tách bằng dấu chấm phẩy (;). Bắt đầu từ 1 và dừng khi i <= 5 sẽ lấy đủ 1..5.",
                "explanation": "Cú pháp chuẩn của for dùng dấu ';' để ngăn cách khởi tạo, điều kiện lặp và bước nhảy. Đáp án B sai vì dùng dấu phẩy, C chỉ in 1..4, D in 0..5.",
            },
            {
                "topic": "Khối 3: Cấu trúc điều khiển rẽ nhánh & Vòng lặp",
                "difficulty": "easy",
                "question_type": "guided_basic",
                "question": "Trong cấu trúc switch-case của C, lệnh break có tác dụng gì?",
                "code_snippet": "switch (chon) {\n    case 1: printf(\"Mot\"); break;\n    default: printf(\"Khac\");\n}",
                "options": {
                    "A": "Thoát khỏi cấu trúc switch sau khi thực hiện xong nhánh case phù hợp",
                    "B": "Kết thúc toàn bộ chương trình main() ngay lập tức",
                    "C": "Quay lại kiểm tra từ case 1",
                    "D": "Bỏ qua lỗi cú pháp nếu thiếu dấu hai chấm",
                },
                "correct_answer": "A",
                "hint": "Gợi ý từng bước: Nếu thiếu break, chương trình sẽ trôi thẳng xuống các nhánh case bên dưới (fall-through).",
                "explanation": "Lệnh break ngắt luồng chạy bên trong switch-case để không thực hiện nhầm các nhánh case kế tiếp.",
            },
        ],
        "bug_trace": [
            {
                "topic": "Khối 3: Cấu trúc điều khiển rẽ nhánh & Vòng lặp",
                "difficulty": "medium",
                "question_type": "bug_analysis",
                "question": "Đoạn mã C kiểm tra điểm đạt môn dưới đây có lỗi logic gì?",
                "code_snippet": "int diem = 3;\nif (diem >= 5);\n    printf(\"Qua mon\\n\");",
                "options": {
                    "A": "Có dấu chấm phẩy (;) ngay sau if (diem >= 5) nên luôn in ra 'Qua mon'",
                    "B": "Biến diem chưa được khởi tạo giá trị",
                    "C": "Toán tử >= không dùng được cho kiểu int",
                    "D": "Không có lỗi gì, chương trình không in ra gì cả",
                },
                "correct_answer": "A",
                "hint": "Quan sát kỹ ký tự kết thúc ngay sau dấu ngoặc đóng của lệnh if.",
                "explanation": "Lần vết: dấu ';' ngay sau if (diem >= 5); tạo thành một câu lệnh rỗng làm thân của if. Lệnh printf(\"Qua mon\\n\"); nằm ngoài if nên luôn thực thi dù diem = 3 < 5.",
            },
            {
                "topic": "Khối 3: Cấu trúc điều khiển rẽ nhánh & Vòng lặp",
                "difficulty": "medium",
                "question_type": "code_trace",
                "question": "Đoạn code C sau in ra màn hình giá trị của biến sum là bao nhiêu?",
                "code_snippet": "int sum = 0;\nfor (int i = 1; i <= 5; i++) {\n    if (i % 2 == 0) continue;\n    sum += i;\n}\nprintf(\"%d\", sum);",
                "options": {"A": "6", "B": "9", "C": "15", "D": "10"},
                "correct_answer": "B",
                "hint": "Lệnh continue bỏ qua phần thân còn lại của vòng lặp khi i là số chẵn.",
                "explanation": "Lần vết: i=1 (lẻ) -> sum=1; i=2 (chẵn) -> continue; i=3 (lẻ) -> sum=4; i=4 (chẵn) -> continue; i=5 (lẻ) -> sum=9. Kết quả in ra là 9.",
            },
        ],
        "standard": [
            {
                "topic": "Khối 3: Cấu trúc điều khiển rẽ nhánh & Vòng lặp",
                "difficulty": "medium",
                "question_type": "code_trace",
                "question": "Trong đoạn mã sau, chữ 'Hi' được in ra bao nhiêu lần?",
                "code_snippet": "int n = 0;\nwhile (n > 0) {\n    printf(\"Hi\");\n    n--;\n}",
                "options": {"A": "0 lần", "B": "1 lần", "C": "Vô hạn lần", "D": "Lỗi biên dịch"},
                "correct_answer": "A",
                "hint": "Kiểm tra điều kiện n > 0 ngay ở lần lặp đầu tiên khi n = 0.",
                "explanation": "Với n = 0, biểu thức điều kiện (0 > 0) nhận giá trị sai ngay từ đầu nên vòng lặp while không thực hiện lần nào.",
            },
        ],
    },
    "IT1108_KB04": {
        "guided_basic": [
            {
                "topic": "Khối 4: Hàm & Mảng (1 chiều, nhiều chiều)",
                "difficulty": "easy",
                "question_type": "guided_basic",
                "question": "Với khai báo mảng int a[5] = {10, 20, 30, 40, 50}; trong ngôn ngữ C, chỉ số hợp lệ của mảng nằm trong khoảng nào?",
                "code_snippet": "int a[5] = {10, 20, 30, 40, 50};",
                "options": {"A": "Từ 0 đến 4", "B": "Từ 1 đến 5", "C": "Từ 0 đến 5", "D": "Từ 1 đến 4"},
                "correct_answer": "A",
                "hint": "Gợi ý từng bước: Mảng trong C luôn đánh chỉ số bắt đầu từ 0, nên phần tử cuối của mảng n phần tử nằm ở vị trí n - 1.",
                "explanation": "Mảng 5 phần tử đánh chỉ số a[0]..a[4]. Truy cập a[5] là vượt quá giới hạn mảng (Out of bounds).",
            },
        ],
        "bug_trace": [
            {
                "topic": "Khối 4: Hàm & Mảng (1 chiều, nhiều chiều)",
                "difficulty": "medium",
                "question_type": "bug_analysis",
                "question": "Đoạn code dùng hàm hoan_vi để đổi chỗ hai số dưới đây in ra kết quả gì tại main() và tại sao?",
                "code_snippet": "#include <stdio.h>\nvoid hoan_vi(int x, int y) {\n    int tmp = x;\n    x = y;\n    y = tmp;\n}\nint main() {\n    int a = 3, b = 7;\n    hoan_vi(a, b);\n    printf(\"%d %d\", a, b);\n    return 0;\n}",
                "options": {
                    "A": "7 3",
                    "B": "3 7 (vì hàm truyền tham trị nên không đổi giá trị biến gốc)",
                    "C": "3 3",
                    "D": "7 7",
                },
                "correct_answer": "B",
                "hint": "Theo dõi phạm vi biến: x, y là bản sao cục bộ trong hoan_vi hay trỏ trực tiếp vào vùng nhớ của a, b?",
                "explanation": "Lần vết: khi gọi hoan_vi(a, b), C truyền theo tham trị (pass-by-value), tạo hai bản sao x=3, y=7. Đổi chỗ x và y chỉ thay đổi bản sao cục bộ; khi thoát hàm, a=3 và b=7 ở main() giữ nguyên.",
            },
            {
                "topic": "Khối 4: Hàm & Mảng (1 chiều, nhiều chiều)",
                "difficulty": "medium",
                "question_type": "code_trace",
                "question": "Cho hàm đệ quy sau. Giá trị trả về của f(4) là bao nhiêu?",
                "code_snippet": "int f(int n) {\n    if (n <= 1) return 1;\n    return n + f(n - 1);\n}",
                "options": {"A": "10", "B": "24", "C": "9", "D": "4"},
                "correct_answer": "A",
                "hint": "Triển khai từng bước từ f(4) xuống điều kiện dừng n <= 1 rồi cộng ngược lên.",
                "explanation": "Lần vết lời gọi đệ quy: f(4) = 4 + f(3) = 4 + 3 + f(2) = 4 + 3 + 2 + f(1). Với n=1 trả về 1. Tổng f(4) = 4 + 3 + 2 + 1 = 10.",
            },
            {
                "topic": "Khối 4: Hàm & Mảng (1 chiều, nhiều chiều)",
                "difficulty": "medium",
                "question_type": "bug_analysis",
                "question": "Vòng lặp duyệt mảng sau bị mắc lỗi phổ biến nào khi tính tổng 4 phần tử của mảng a?",
                "code_snippet": "int a[4] = {2, 4, 6, 8};\nint s = 0;\nfor (int i = 0; i <= 4; i++) {\n    s += a[i];\n}",
                "options": {
                    "A": "Lỗi tràn chỉ số mảng (off-by-one) vì truy cập a[4] nằm ngoài mảng",
                    "B": "Không thể khai báo int i bên trong vòng for",
                    "C": "Biến s phải khởi tạo bằng 1",
                    "D": "Mảng a chỉ có 3 phần tử",
                },
                "correct_answer": "A",
                "hint": "Kiểm tra điều kiện dừng i <= 4 khi kích thước mảng là 4.",
                "explanation": "Lần vết: mảng a[4] chỉ có các chỉ số 0, 1, 2, 3. Điều kiện i <= 4 khiến vòng lặp chạy thêm bước i=4 và đọc vùng nhớ ngoài mảng. Điều kiện đúng là i < 4.",
            },
        ],
        "standard": [
            {
                "topic": "Khối 4: Hàm & Mảng (1 chiều, nhiều chiều)",
                "difficulty": "medium",
                "question_type": "concept",
                "question": "Trong hàm đệ quy, 'trường hợp cơ sở' (base case) dùng để làm gì?",
                "code_snippet": "",
                "options": {
                    "A": "Dừng chuỗi gọi đệ quy để tránh tràn bộ nhớ ngăn xếp (Stack Overflow)",
                    "B": "Tăng tốc độ biên dịch chương trình",
                    "C": "Chuyển mảng 1 chiều thành 2 chiều",
                    "D": "Khởi tạo biến toàn cục",
                },
                "correct_answer": "A",
                "hint": "Điều gì xảy ra nếu hàm tự gọi lại chính nó mà không có điểm dừng?",
                "explanation": "Mỗi hàm đệ quy bắt buộc có điều kiện dừng (base case); nếu thiếu, hàm gọi mãi cho đến khi hết bộ nhớ Stack.",
            },
        ],
    },
    "IT1108_KB05": {
        "guided_basic": [
            {
                "topic": "Khối 5: Con trỏ, Bộ nhớ động & Xâu ký tự",
                "difficulty": "easy",
                "question_type": "guided_basic",
                "question": "Trong ngôn ngữ C, xâu ký tự (string) kết thúc bằng ký tự đặc biệt nào?",
                "code_snippet": "char s[] = \"IT1108\";",
                "options": {"A": "'\\0' (ký tự null)", "B": "'\\n' (xuống dòng)", "C": "' ' (dấu cách)", "D": "EOF"},
                "correct_answer": "A",
                "hint": "Gợi ý từng bước: Chuỗi 'IT1108' có 6 ký tự nhưng mảng char s[] chiếm 7 byte vì thêm ký tự '\\0' ở cuối.",
                "explanation": "Trong C, xâu ký tự là mảng char kết thúc bằng ký tự null '\\0' để các hàm như strlen(), printf(\"%s\") biết điểm kết thúc chuỗi.",
            },
            {
                "topic": "Khối 5: Con trỏ, Bộ nhớ động & Xâu ký tự",
                "difficulty": "easy",
                "question_type": "guided_basic",
                "question": "Toán tử nào dùng để lấy địa chỉ vùng nhớ của một biến trong ngôn ngữ C?",
                "code_snippet": "int n = 10;\nint *p = /* lấy địa chỉ của n */;",
                "options": {"A": "&n", "B": "*n", "C": "->n", "D": "addr(n)"},
                "correct_answer": "A",
                "hint": "Gợi ý từng bước: Toán tử & trả về địa chỉ của biến, còn toán tử * dùng để khai báo con trỏ hoặc truy xuất giá trị tại địa chỉ.",
                "explanation": "Toán tử & (address-of) trả về địa chỉ bộ nhớ của biến n để gán cho con trỏ int *p = &n.",
            },
        ],
        "bug_trace": [
            {
                "topic": "Khối 5: Con trỏ, Bộ nhớ động & Xâu ký tự",
                "difficulty": "medium",
                "question_type": "code_trace",
                "question": "Sau khi thực thi đoạn mã con trỏ dưới đây, giá trị của x và *p lần lượt là bao nhiêu?",
                "code_snippet": "int x = 10;\nint *p = &x;\n*p = *p + 5;",
                "options": {
                    "A": "x = 15, *p = 15",
                    "B": "x = 10, *p = 15",
                    "C": "x = 15, *p = 10",
                    "D": "Lỗi vì không thể cộng với *p",
                },
                "correct_answer": "A",
                "hint": "p đang lưu địa chỉ của x, nên *p truy cập trực tiếp vào vùng nhớ của x.",
                "explanation": "Lần vết: p = &x làm p trỏ vào biến x. Biểu thức *p = *p + 5 đọc giá trị tại địa chỉ x (10), cộng 5 thành 15 và ghi lại vào x. Do đó cả x và *p đều bằng 15.",
            },
            {
                "topic": "Khối 5: Con trỏ, Bộ nhớ động & Xâu ký tự",
                "difficulty": "medium",
                "question_type": "bug_analysis",
                "question": "Đoạn code cấp phát động mảng 5 số nguyên sau thiếu thao tác quan trọng nào để tránh rò rỉ bộ nhớ?",
                "code_snippet": "#include <stdlib.h>\nint main() {\n    int *arr = (int*)malloc(5 * sizeof(int));\n    if (arr == NULL) return 1;\n    arr[0] = 9;\n    return 0;\n}",
                "options": {
                    "A": "Chưa gọi free(arr); trước khi kết thúc chương trình",
                    "B": "Thay sizeof(int) bằng sizeof(char)",
                    "C": "Phải dùng scanf để cấp phát bộ nhớ",
                    "D": "Kiểm tra arr == NULL là thừa",
                },
                "correct_answer": "A",
                "hint": "Vùng nhớ cấp phát trên Heap bằng malloc() có tự động thu hồi như biến cục bộ không?",
                "explanation": "Bộ nhớ xin qua malloc/calloc nằm trên Heap nên cần được giải phóng bằng free(arr); khi dùng xong để tránh Memory Leak.",
            },
        ],
        "standard": [
            {
                "topic": "Khối 5: Con trỏ, Bộ nhớ động & Xâu ký tự",
                "difficulty": "hard",
                "question_type": "code_trace",
                "question": "Kết quả in ra của đoạn mã con trỏ và mảng sau là gì?",
                "code_snippet": "int a[] = {4, 8, 12, 16};\nint *p = a;\nprintf(\"%d\", *(p + 2));",
                "options": {"A": "6", "B": "12", "C": "8", "D": "16"},
                "correct_answer": "B",
                "hint": "Tên mảng a trỏ tới a[0]; phép cộng con trỏ p + 2 dịch chuyển tới phần tử chỉ số 2.",
                "explanation": "p trỏ tới a[0]=4, p+1 trỏ tới a[1]=8, p+2 trỏ tới a[2]=12. Toán tử giải tham chiếu *(p + 2) trả về giá trị 12.",
            },
        ],
    },
    "IT1108_KB06": {
        "guided_basic": [
            {
                "topic": "Khối 6: Cấu trúc (struct), Tệp dữ liệu & Ôn tập cuối kỳ",
                "difficulty": "easy",
                "question_type": "guided_basic",
                "question": "Để mở một tệp văn bản 'data.txt' chỉ phục vụ đọc dữ liệu trong C, ta truyền chế độ (mode) nào vào hàm fopen?",
                "code_snippet": "FILE *f = fopen(\"data.txt\", /* mode */);",
                "options": {"A": "\"r\"", "B": "\"w\"", "C": "\"wb\"", "D": "\"a\""},
                "correct_answer": "A",
                "hint": "Gợi ý từng bước: 'r' viết tắt của read (đọc), 'w' là write (ghi đè mới), 'a' là append (ghi nối thêm).",
                "explanation": "Chế độ \"r\" mở tệp văn bản để đọc (trả về NULL nếu tệp chưa tồn tại). Dùng \"w\" khi muốn đọc sẽ làm xóa trắng dữ liệu tệp cũ.",
            },
            {
                "topic": "Khối 6: Cấu trúc (struct), Tệp dữ liệu & Ôn tập cuối kỳ",
                "difficulty": "easy",
                "question_type": "guided_basic",
                "question": "Sau khi đọc/ghi xong một tệp đã mở qua biến FILE *f, ta cần gọi hàm nào để đóng tệp an toàn?",
                "code_snippet": "FILE *f = fopen(\"out.txt\", \"w\");\n// thao tác ghi dữ liệu\n/* đóng tệp */",
                "options": {"A": "fclose(f);", "B": "free(f);", "C": "closefile(f);", "D": "end(f);"},
                "correct_answer": "A",
                "hint": "Gợi ý từng bước: Cặp hàm chuẩn trong <stdio.h> để mở và đóng tệp là fopen(...) và fclose(...).",
                "explanation": "Hàm fclose(f) đẩy toàn bộ dữ liệu từ bộ đệm xuống đĩa và giải phóng tài nguyên tệp đang mở.",
            },
        ],
        "bug_trace": [
            {
                "topic": "Khối 6: Cấu trúc (struct), Tệp dữ liệu & Ôn tập cuối kỳ",
                "difficulty": "medium",
                "question_type": "bug_analysis",
                "question": "Trong hàm sử dụng con trỏ cấu trúc bên dưới, lỗi cú pháp nào xảy ra ở dòng gán điểm?",
                "code_snippet": "struct SinhVien { float diem; };\nvoid dat_diem(struct SinhVien *sv) {\n    sv.diem = 8.5;\n}",
                "options": {
                    "A": "sv là con trỏ struct nên phải dùng toán tử mũi tên sv->diem thay vì sv.diem",
                    "B": "Kiểu float không được đặt trong struct",
                    "C": "Thiếu dấu chấm phẩy sau float diem",
                    "D": "Hàm void không nhận được con trỏ struct",
                },
                "correct_answer": "A",
                "hint": "Phân biệt truy cập thành viên qua biến struct (dấu chấm .) và qua con trỏ struct (dấu mũi tên ->).",
                "explanation": "Với biến struct dùng sv.diem; nhưng vì tham số sv là con trỏ (struct SinhVien *sv), bắt buộc dùng sv->diem (hoặc (*sv).diem).",
            },
            {
                "topic": "Khối 6: Cấu trúc (struct), Tệp dữ liệu & Ôn tập cuối kỳ",
                "difficulty": "medium",
                "question_type": "code_trace",
                "question": "Kết quả in ra màn hình của đoạn mã sử dụng struct sau là gì?",
                "code_snippet": "typedef struct { int x; int y; } Diem;\nDiem p = {3, 4};\nprintf(\"%d\", p.x + p.y);",
                "options": {"A": "7", "B": "12", "C": "34", "D": "Lỗi biên dịch"},
                "correct_answer": "A",
                "hint": "Thứ tự giá trị khởi tạo trong {3, 4} tương ứng lần lượt với trường x và trường y của struct Diem.",
                "explanation": "Lần vết: p.x = 3, p.y = 4. Biểu thức p.x + p.y cho kết quả 3 + 4 = 7.",
            },
        ],
        "standard": [
            {
                "topic": "Khối 6: Cấu trúc (struct), Tệp dữ liệu & Ôn tập cuối kỳ",
                "difficulty": "hard",
                "question_type": "concept",
                "question": "Cặp hàm nào trong <stdio.h> được thiết kế để đọc/ghi khối dữ liệu cấu trúc trên tệp nhị phân?",
                "code_snippet": "",
                "options": {
                    "A": "fread() và fwrite()",
                    "B": "fscanf() và fprintf()",
                    "C": "fgets() và fputs()",
                    "D": "scanf() và printf()",
                },
                "correct_answer": "A",
                "hint": "Các hàm làm việc với tệp nhị phân (\"rb\", \"wb\") xử lý trực tiếp theo kích thước byte và số phần tử.",
                "explanation": "fread() và fwrite() đọc/ghi trực tiếp khối byte nhị phân (như struct hoặc mảng) với hiệu năng cao và không cần chuyển đổi văn bản.",
            },
        ],
    },
}


def _pick_variant_for_block(
    block_info: Dict[str, Any],
    scenario_type: str,
    requested_difficulty: str,
    used_texts: set,
) -> Dict[str, Any]:
    block_code = block_info["it1108_block_code"]
    status = block_info.get("submission_vs_score_status", "LOW_SCORE_DESPITE_SUBMISSION")
    bank = _QUESTION_BANK.get(block_code, _QUESTION_BANK["IT1108_KB03"])

    if scenario_type == "baseline":
        priority_buckets = ["standard", "bug_trace", "guided_basic"]
    elif scenario_type == "risk_level":
        priority_buckets = ["guided_basic", "standard", "bug_trace"]
    else:
        if status == "MISSING_SUBMISSION":
            priority_buckets = ["guided_basic", "standard", "bug_trace"]
        elif status == "LOW_SCORE_DESPITE_SUBMISSION":
            priority_buckets = ["bug_trace", "standard", "guided_basic"]
        elif status == "PARTIAL_SUBMISSION_AND_LOW_SCORE":
            priority_buckets = ["guided_basic", "bug_trace", "standard"]
        elif status == "NORMAL":
            priority_buckets = ["standard", "bug_trace", "guided_basic"]
        else:
            priority_buckets = ["standard", "guided_basic", "bug_trace"]

    chosen: Optional[Dict[str, Any]] = None
    for bucket in priority_buckets:
        for candidate in bank.get(bucket, []):
            if candidate["question"] not in used_texts:
                chosen = dict(candidate)
                chosen["it1108_block_code"] = block_code
                break
        if chosen is not None:
            break

    if chosen is None:
        # Dự phòng nếu yêu cầu > 5 câu trên cùng 1 khối
        for bucket in ("guided_basic", "bug_trace", "standard"):
            if bank.get(bucket):
                chosen = dict(bank[bucket][0])
                chosen["it1108_block_code"] = block_code
                break

    assert chosen is not None
    used_texts.add(chosen["question"])

    if scenario_type == "baseline":
        chosen["difficulty"] = requested_difficulty if requested_difficulty != "auto" else "medium"
        if chosen["question_type"] == "guided_basic":
            chosen["question_type"] = "concept"
        chosen["hint"] = "Xem lại giáo trình bài giảng tương ứng."
        chosen["explanation"] = chosen["explanation"].split(".")[0].strip() + "."
    elif requested_difficulty != "auto":
        chosen["difficulty"] = requested_difficulty

    return chosen


def generate_mock_quiz(
    profile: Dict[str, Any],
    scenario_type: str,
    prompt_bundle: Dict[str, Any],
    num_questions: int = 5,
    difficulty: str = "auto",
    simulate_error: Optional[str] = None,
) -> Tuple[str, Dict[str, int]]:
    """
    Sinh chuỗi JSON (hoặc kích hoạt lỗi mô phỏng) cho chế độ Mock Offline.
    Trả về (raw_text, token_usage).
    """
    if simulate_error == "timeout":
        raise socket.timeout("Mô phỏng Timeout sau 20s (NFR-01).")
    if simulate_error == "rate_limit":
        raise MockRateLimitError("429 RESOURCE_EXHAUSTED: Mô phỏng vượt ngưỡng lượt gọi miễn phí Gemini API.")
    if simulate_error == "service_unavailable":
        raise MockServiceUnavailableError(
            "HTTP 503 UNAVAILABLE: Mô phỏng dịch vụ Google Gemini Free Tier đang quá tải (high demand)."
        )
    if simulate_error == "malformed_json":
        return (
            "Xin lỗi, đây là bài tập môn C nhưng không ở định dạng JSON hợp lệ: {broken_json: [1, 2",
            {"prompt_tokens": 210, "output_tokens": 32},
        )
    if simulate_error == "schema_error":
        return (
            json.dumps(
                {
                    "course_code": "IT1108",
                    "course_name": "Nhập môn lập trình",
                    "scenario_type": scenario_type,
                    "questions": [{"question_number": 1, "question": "Thiếu các trường bắt buộc"}],
                },
                ensure_ascii=False,
            ),
            {"prompt_tokens": 210, "output_tokens": 55},
        )

    req_params = prompt_bundle.get("request_params") or {}
    course_code = req_params.get("course_code", "IT1108")
    course_name = req_params.get("course_name", "Nhập môn lập trình")
    review_topics = list(req_params.get("review_topics") or [])
    req_diff = str(difficulty or req_params.get("difficulty") or "auto").lower()

    sanitized = prompt_bundle.get("sanitized_profile", {})
    blocks: List[Dict[str, Any]] = sanitized.get("weak_blocks", [])
    if not blocks:
        blocks = [
            {
                "it1108_block_code": "IT1108_KB03",
                "it1108_block_name": "Khối 3: Cấu trúc điều khiển rẽ nhánh & Vòng lặp",
                "submission_vs_score_status": "MISSING_SUBMISSION",
            }
        ]

    risk_level = str(sanitized.get("risk_level", "MEDIUM")).upper()
    risk_prob = float(sanitized.get("risk_probability", 0.50))

    questions: List[Dict[str, Any]] = []
    used_texts = set()
    for idx in range(num_questions):
        target_block = blocks[idx % len(blocks)]
        q_item = _pick_variant_for_block(target_block, scenario_type, req_diff, used_texts)
        q_item["question_number"] = idx + 1
        if review_topics and len(blocks) == 1:
            q_item["topic"] = f"{target_block['it1108_block_name']} ({review_topics[idx % len(review_topics)]})"
        questions.append(q_item)

    if scenario_type == "baseline":
        difficulty_strategy = "Phân bổ đề ôn tập chung theo chủ đề học phần, chưa xét đến ngữ cảnh mô phỏng của người học."
        study_advice = "Ôn tập lại kiến thức trọng tâm các chủ đề đã chọn và luyện tập thêm bài tập mẫu."
    elif scenario_type == "risk_level":
        difficulty_strategy = (
            f"Điều chỉnh độ khó tổng thể theo mức rủi ro tham khảo {risk_level} (xác suất tham khảo {risk_prob:.2f}): "
            "ưu tiên củng cố kiến thức cốt lõi trước khi làm câu hỏi vận dụng."
        )
        study_advice = (
            f"Mức tham khảo hiện tại là {risk_level}. Bạn nên làm đủ {num_questions} câu ôn tập và đọc kỹ phần giải thích để củng cố kiến thức."
        )
    else:
        has_missing = any(b.get("submission_vs_score_status") == "MISSING_SUBMISSION" for b in blocks)
        has_low_score = any(b.get("submission_vs_score_status") == "LOW_SCORE_DESPITE_SUBMISSION" for b in blocks)
        strategy_parts = []
        if has_missing:
            strategy_parts.append("ưu tiên câu easy dạng guided_basic kèm gợi ý từng bước cho chủ đề chưa có bài nộp")
        if has_low_score:
            strategy_parts.append("tăng câu code_trace/bug_analysis kèm giải thích lần vết từng dòng cho chủ đề điểm còn thấp")
        if not strategy_parts:
            strategy_parts.append("tập trung câu hỏi vận dụng medium/hard để củng cố kiến thức")
        difficulty_strategy = f"Cá nhân hóa tham khảo (rủi ro {risk_level}, p={risk_prob:.2f}): " + "; ".join(strategy_parts) + "."
        study_advice = (
            "Gợi ý lộ trình ôn tập: hoàn thành các câu hướng dẫn từng bước trước ở phần chưa nộp bài, "
            "sau đó tự lần vết trên giấy các đoạn code C để khắc phục lỗi sai thường gặp."
        )

    payload = {
        "course_code": course_code,
        "course_name": course_name,
        "scenario_type": scenario_type,
        "target_blocks": [b["it1108_block_code"] for b in blocks],
        "review_topics": review_topics or [b["it1108_block_name"] for b in blocks],
        "difficulty_strategy": difficulty_strategy,
        "study_advice": study_advice,
        "questions": questions,
    }
    raw_json = json.dumps(payload, ensure_ascii=False, indent=2)
    prompt_tokens = max(120, len(prompt_bundle.get("user_prompt", "")) // 4)
    output_tokens = max(250, len(raw_json) // 4)
    return raw_json, {"prompt_tokens": prompt_tokens, "output_tokens": output_tokens}
