-- =====================================================================================
-- DỰ ÁN: SMART LMS & GENERATIVE AI
-- TỆP: create_tables.sql
-- MỤC ĐÍCH: KHỞI TẠO CƠ SỞ DỮ LIỆU VÀ CẤU TRÚC LƯU TRỮ TRÊN MICROSOFT SQL SERVER
-- KẾ THỪA THIẾT KẾ BÁO CÁO TUẦN 2 VÀ BỔ SUNG BẢNG ĐẶC TRƯNG TIỀN XỬ LÝ
-- =====================================================================================

USE master;
GO

-- 1. Tạo cơ sở dữ liệu nếu chưa tồn tại
IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = N'SmartLMS_DB')
BEGIN
    CREATE DATABASE SmartLMS_DB;
    PRINT N'Đã khởi tạo cơ sở dữ liệu SmartLMS_DB.';
END
GO

USE SmartLMS_DB;
GO

-- 2. XÓA BẢNG CŨ NẾU ĐÃ TỒN TẠI (THEO THỨ TỰ RÀNG BUỘC KHÓA NGOẠI)
IF OBJECT_ID(N'dbo.ai_quizzes', N'U') IS NOT NULL DROP TABLE dbo.ai_quizzes;
IF OBJECT_ID(N'dbo.risk_predictions', N'U') IS NOT NULL DROP TABLE dbo.risk_predictions;
IF OBJECT_ID(N'dbo.app_users', N'U') IS NOT NULL DROP TABLE dbo.app_users;
IF OBJECT_ID(N'dbo.student_features', N'U') IS NOT NULL DROP TABLE dbo.student_features;
IF OBJECT_ID(N'dbo.studentVle', N'U') IS NOT NULL DROP TABLE dbo.studentVle;
IF OBJECT_ID(N'dbo.studentAssessment', N'U') IS NOT NULL DROP TABLE dbo.studentAssessment;
IF OBJECT_ID(N'dbo.studentRegistration', N'U') IS NOT NULL DROP TABLE dbo.studentRegistration;
IF OBJECT_ID(N'dbo.studentInfo', N'U') IS NOT NULL DROP TABLE dbo.studentInfo;
IF OBJECT_ID(N'dbo.vle', N'U') IS NOT NULL DROP TABLE dbo.vle;
IF OBJECT_ID(N'dbo.assessments', N'U') IS NOT NULL DROP TABLE dbo.assessments;
IF OBJECT_ID(N'dbo.courses', N'U') IS NOT NULL DROP TABLE dbo.courses;
GO

-- =====================================================================================
-- PHẦN I. CÁC BẢNG DỮ LIỆU GỐC OULAD (RAW / HISTORY TABLES)
-- =====================================================================================

-- Bảng 1: courses
CREATE TABLE dbo.courses (
    code_module VARCHAR(10) NOT NULL,
    code_presentation VARCHAR(10) NOT NULL,
    module_presentation_length INT NOT NULL,
    CONSTRAINT PK_courses PRIMARY KEY CLUSTERED (code_module, code_presentation)
);
GO

-- Bảng 2: assessments
CREATE TABLE dbo.assessments (
    id_assessment INT NOT NULL,
    code_module VARCHAR(10) NOT NULL,
    code_presentation VARCHAR(10) NOT NULL,
    assessment_type VARCHAR(10) NOT NULL,
    [date] INT NULL,
    [weight] FLOAT NOT NULL,
    CONSTRAINT PK_assessments PRIMARY KEY CLUSTERED (id_assessment),
    CONSTRAINT FK_assessments_courses FOREIGN KEY (code_module, code_presentation)
        REFERENCES dbo.courses (code_module, code_presentation)
);
GO

-- Bảng 3: vle
CREATE TABLE dbo.vle (
    id_site INT NOT NULL,
    code_module VARCHAR(10) NOT NULL,
    code_presentation VARCHAR(10) NOT NULL,
    activity_type VARCHAR(50) NOT NULL,
    week_from INT NULL,
    week_to INT NULL,
    CONSTRAINT PK_vle PRIMARY KEY CLUSTERED (id_site),
    CONSTRAINT FK_vle_courses FOREIGN KEY (code_module, code_presentation)
        REFERENCES dbo.courses (code_module, code_presentation)
);
GO

-- Bảng 4: studentInfo
CREATE TABLE dbo.studentInfo (
    code_module VARCHAR(10) NOT NULL,
    code_presentation VARCHAR(10) NOT NULL,
    id_student INT NOT NULL,
    gender VARCHAR(5) NOT NULL,
    region VARCHAR(100) NOT NULL,
    highest_education VARCHAR(100) NOT NULL,
    imd_band VARCHAR(20) NULL,
    age_band VARCHAR(20) NOT NULL,
    num_of_prev_attempts INT NOT NULL,
    studied_credits INT NOT NULL,
    disability VARCHAR(5) NOT NULL,
    final_result VARCHAR(20) NOT NULL,
    CONSTRAINT PK_studentInfo PRIMARY KEY CLUSTERED (code_module, code_presentation, id_student),
    CONSTRAINT FK_studentInfo_courses FOREIGN KEY (code_module, code_presentation)
        REFERENCES dbo.courses (code_module, code_presentation)
);
GO

-- Bảng 5: studentRegistration
CREATE TABLE dbo.studentRegistration (
    code_module VARCHAR(10) NOT NULL,
    code_presentation VARCHAR(10) NOT NULL,
    id_student INT NOT NULL,
    date_registration INT NULL,
    date_unregistration INT NULL,
    CONSTRAINT PK_studentRegistration PRIMARY KEY CLUSTERED (code_module, code_presentation, id_student),
    CONSTRAINT FK_studentRegistration_courses FOREIGN KEY (code_module, code_presentation)
        REFERENCES dbo.courses (code_module, code_presentation)
);
GO

-- Bảng 6: studentAssessment
CREATE TABLE dbo.studentAssessment (
    id_assessment INT NOT NULL,
    id_student INT NOT NULL,
    date_submitted INT NOT NULL,
    is_banked INT NOT NULL,
    score FLOAT NULL,
    CONSTRAINT PK_studentAssessment PRIMARY KEY CLUSTERED (id_assessment, id_student),
    CONSTRAINT FK_studentAssessment_assessments FOREIGN KEY (id_assessment)
        REFERENCES dbo.assessments (id_assessment)
);
GO

-- Bảng 7: studentVle (Dung lượng lớn: ~10.6 triệu dòng)
CREATE TABLE dbo.studentVle (
    code_module VARCHAR(10) NOT NULL,
    code_presentation VARCHAR(10) NOT NULL,
    id_student INT NOT NULL,
    id_site INT NOT NULL,
    [date] INT NOT NULL,
    sum_click INT NOT NULL
);
GO

-- Tạo Clustered Index tối ưu truy vấn theo sinh viên và môn học
CREATE CLUSTERED INDEX CIX_studentVle ON dbo.studentVle (code_module, code_presentation, id_student, [date]);
CREATE NONCLUSTERED INDEX NIX_studentVle_site ON dbo.studentVle (id_site) INCLUDE (sum_click);
GO

-- =====================================================================================
-- PHẦN II. BẢNG ĐẶC TRƯNG HỌC TẬP TIỀN XỬ LÝ (FEATURE STORE TABLE)
-- PHỤC VỤ TRUY VẤN SUY LUẬN NHANH (INFERENCE) CỦA MÔ HÌNH RANDOM FOREST
-- =====================================================================================

CREATE TABLE dbo.student_features (
    id_student INT NOT NULL,
    code_module VARCHAR(10) NOT NULL,
    code_presentation VARCHAR(10) NOT NULL,
    
    -- Nhóm đặc trưng điểm số (Group A)
    num_assessments INT NOT NULL DEFAULT 0,
    assessment_completion_rate FLOAT NOT NULL DEFAULT 0.0,
    avg_score FLOAT NOT NULL DEFAULT 0.0,
    weighted_score FLOAT NOT NULL DEFAULT 0.0,
    min_score FLOAT NOT NULL DEFAULT 0.0,
    max_score FLOAT NOT NULL DEFAULT 0.0,
    score_std FLOAT NOT NULL DEFAULT 0.0,
    tma_avg_score FLOAT NOT NULL DEFAULT 0.0,
    cma_avg_score FLOAT NOT NULL DEFAULT 0.0,
    late_submission_count INT NOT NULL DEFAULT 0,
    avg_submission_delay FLOAT NOT NULL DEFAULT 0.0,

    -- Nhóm đặc trưng hành vi VLE (Group B)
    total_clicks INT NOT NULL DEFAULT 0,
    active_days INT NOT NULL DEFAULT 0,
    average_clicks_per_day FLOAT NOT NULL DEFAULT 0.0,
    unique_learning_resources INT NOT NULL DEFAULT 0,
    early_clicks_pre_course INT NOT NULL DEFAULT 0,
    clicks_first_half INT NOT NULL DEFAULT 0,
    clicks_oucontent INT NOT NULL DEFAULT 0,
    clicks_forumng INT NOT NULL DEFAULT 0,
    clicks_quiz INT NOT NULL DEFAULT 0,
    clicks_resource_url INT NOT NULL DEFAULT 0,

    -- Nhóm đặc trưng học tập & nhân khẩu học (Group C)
    num_of_prev_attempts INT NOT NULL DEFAULT 0,
    studied_credits INT NOT NULL DEFAULT 0,
    date_registration FLOAT NOT NULL DEFAULT 0.0,
    has_unregistered INT NOT NULL DEFAULT 0,
    gender VARCHAR(5) NULL,
    highest_education VARCHAR(100) NULL,
    imd_band VARCHAR(20) NULL,
    age_band VARCHAR(20) NULL,
    disability VARCHAR(5) NULL,

    -- Biến mục tiêu (Target)
    final_result VARCHAR(20) NOT NULL,
    is_at_risk INT NOT NULL, -- 1: At-Risk (Fail/Withdrawn), 0: Non-Risk (Pass/Distinction)

    CONSTRAINT PK_student_features PRIMARY KEY CLUSTERED (code_module, code_presentation, id_student)
);
GO

-- =====================================================================================
-- PHẦN III. BẢNG ỨNG DỤNG HỆ THỐNG SMART LMS (KẾ THỪA THIẾT KẾ TUẦN 2)
-- =====================================================================================

CREATE TABLE dbo.app_users (
    user_id INT IDENTITY(1,1) NOT NULL,
    oulad_student_id INT NULL,
    username VARCHAR(50) NOT NULL UNIQUE,
    full_name NVARCHAR(100) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    created_at DATETIME NOT NULL DEFAULT GETDATE(),
    CONSTRAINT PK_app_users PRIMARY KEY CLUSTERED (user_id)
);
GO

CREATE TABLE dbo.risk_predictions (
    prediction_id INT IDENTITY(1,1) NOT NULL,
    user_id INT NOT NULL,
    code_module VARCHAR(10) NOT NULL,
    code_presentation VARCHAR(10) NOT NULL,
    risk_level VARCHAR(20) NOT NULL, -- High, Medium, Low
    risk_probability FLOAT NOT NULL,
    predicted_at DATETIME NOT NULL DEFAULT GETDATE(),
    CONSTRAINT PK_risk_predictions PRIMARY KEY CLUSTERED (prediction_id),
    CONSTRAINT FK_risk_predictions_users FOREIGN KEY (user_id)
        REFERENCES dbo.app_users (user_id)
);
GO

CREATE TABLE dbo.ai_quizzes (
    quiz_id INT IDENTITY(1,1) NOT NULL,
    user_id INT NOT NULL,
    weak_topic NVARCHAR(200) NOT NULL,
    quiz_content NVARCHAR(MAX) NOT NULL,
    created_at DATETIME NOT NULL DEFAULT GETDATE(),
    CONSTRAINT PK_ai_quizzes PRIMARY KEY CLUSTERED (quiz_id),
    CONSTRAINT FK_ai_quizzes_users FOREIGN KEY (user_id)
        REFERENCES dbo.app_users (user_id),
    CONSTRAINT CK_ai_quizzes_json CHECK (ISJSON(quiz_content) > 0)
);
GO

-- =====================================================================================
-- PHẦN IV. HƯỚNG DẪN BULK INSERT NHẬP DỮ LIỆU DUNG LƯỢNG LỚN VÀO SQL SERVER
-- =====================================================================================
/*
-- 1. Nhập bảng courses:
BULK INSERT dbo.courses
FROM 'C:\Smart_LMS\dataaw\courses.csv'
WITH (FORMAT = 'CSV', FIRSTROW = 2, FIELDTERMINATOR = ',', ROWTERMINATOR = '
', TABLOCK);

-- 2. Nhập bảng assessments:
BULK INSERT dbo.assessments
FROM 'C:\Smart_LMS\dataawssessments.csv'
WITH (FORMAT = 'CSV', FIRSTROW = 2, FIELDTERMINATOR = ',', ROWTERMINATOR = '
', TABLOCK);

-- 3. Nhập bảng vle:
BULK INSERT dbo.vle
FROM 'C:\Smart_LMS\dataawle.csv'
WITH (FORMAT = 'CSV', FIRSTROW = 2, FIELDTERMINATOR = ',', ROWTERMINATOR = '
', TABLOCK);

-- 4. Nhập bảng studentInfo:
BULK INSERT dbo.studentInfo
FROM 'C:\Smart_LMS\dataaw\studentInfo.csv'
WITH (FORMAT = 'CSV', FIRSTROW = 2, FIELDTERMINATOR = ',', ROWTERMINATOR = '
', TABLOCK);

-- 5. Nhập bảng studentRegistration:
BULK INSERT dbo.studentRegistration
FROM 'C:\Smart_LMS\dataaw\studentRegistration.csv'
WITH (FORMAT = 'CSV', FIRSTROW = 2, FIELDTERMINATOR = ',', ROWTERMINATOR = '
', TABLOCK);

-- 6. Nhập bảng studentAssessment:
BULK INSERT dbo.studentAssessment
FROM 'C:\Smart_LMS\dataaw\studentAssessment.csv'
WITH (FORMAT = 'CSV', FIRSTROW = 2, FIELDTERMINATOR = ',', ROWTERMINATOR = '
', TABLOCK);

-- 7. Nhập bảng studentVle (10.6 triệu dòng tối ưu bằng BATCHSIZE):
BULK INSERT dbo.studentVle
FROM 'C:\Smart_LMS\dataaw\studentVle.csv'
WITH (FORMAT = 'CSV', FIRSTROW = 2, FIELDTERMINATOR = ',', ROWTERMINATOR = '
', BATCHSIZE = 100000, TABLOCK);

-- 8. Hoặc sử dụng tiện ích dòng lệnh BCP (Bulk Copy Program) cực nhanh:
-- bcp SmartLMS_DB.dbo.studentVle in "C:\Smart_LMS\dataaw\studentVle.csv" -c -t, -r 
 -F 2 -S localhost -U sa -P YourPassword -b 100000
*/
