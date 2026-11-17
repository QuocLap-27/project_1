-- =====================================================================================
-- DỰ ÁN: SMART LMS & GENERATIVE AI – PHÂN TÍCH DỮ LIỆU NGƯỜI HỌC VÀ TẠO NỘI DUNG TỰ ĐỘNG
-- TỆP: curriculum_mapping_schema.sql
-- MỤC ĐÍCH: THIẾT KẾ CẤU TRÚC BẢNG CURRICULUM/TOPIC MAPPING TRÊN MICROSOFT SQL SERVER
-- TÍCH HỢP VỚI KIẾN TRÚC CƠ SỞ DỮ LIỆU SmartLMS_DB ĐÃ THIẾT KẾ Ở TUẦN 2 & TUẦN 3
-- =====================================================================================

USE SmartLMS_DB;
GO

-- 1. XÓA CÁC VIEW / PROCEDURE / BẢNG MAPPING CŨ NẾU ĐÃ TỒN TẠI
IF OBJECT_ID(N'dbo.usp_GetStudentWeakTopicsForGemini', N'P') IS NOT NULL
    DROP PROCEDURE dbo.usp_GetStudentWeakTopicsForGemini;
GO
IF OBJECT_ID(N'dbo.vw_StudentTopicPerformance', N'V') IS NOT NULL
    DROP VIEW dbo.vw_StudentTopicPerformance;
GO
IF OBJECT_ID(N'dbo.vle_topic_mapping', N'U') IS NOT NULL
    DROP TABLE dbo.vle_topic_mapping;
GO
IF OBJECT_ID(N'dbo.assessment_topic_mapping', N'U') IS NOT NULL
    DROP TABLE dbo.assessment_topic_mapping;
GO
IF OBJECT_ID(N'dbo.curriculum_topics', N'U') IS NOT NULL
    DROP TABLE dbo.curriculum_topics;
GO

-- =====================================================================================
-- BẢNG 1: dbo.curriculum_topics (Danh mục Chủ đề Kiến thức theo Môn học & Học kỳ)
-- =====================================================================================
CREATE TABLE dbo.curriculum_topics (
    topic_id INT NOT NULL,
    code_module VARCHAR(10) NOT NULL,
    code_presentation VARCHAR(10) NOT NULL,
    topic_code VARCHAR(50) NOT NULL,
    topic_name NVARCHAR(255) NOT NULL,
    topic_description NVARCHAR(1000) NULL,
    learning_outcomes NVARCHAR(1000) NULL,
    week_from INT NULL,
    week_to INT NULL,
    source_reference NVARCHAR(500) NOT NULL,
    verification_status VARCHAR(20) NOT NULL DEFAULT 'Pending',
    created_at DATETIME NOT NULL DEFAULT GETDATE(),

    CONSTRAINT PK_curriculum_topics PRIMARY KEY CLUSTERED (topic_id),
    CONSTRAINT UQ_curriculum_topics_code UNIQUE (code_module, code_presentation, topic_code),
    CONSTRAINT FK_curriculum_topics_courses FOREIGN KEY (code_module, code_presentation)
        REFERENCES dbo.courses (code_module, code_presentation),
    CONSTRAINT CK_curriculum_topics_status CHECK (verification_status IN ('Verified', 'Pending', 'Unmapped')),
    CONSTRAINT CK_curriculum_topics_weeks CHECK (week_from IS NULL OR week_to IS NULL OR week_from <= week_to)
);
GO

-- =====================================================================================
-- BẢNG 2: dbo.assessment_topic_mapping (Ánh xạ Bài đánh giá OULAD -> Chủ đề Kiến thức)
-- Cho phép quan hệ N-N (1 bài đánh giá ánh xạ nhiều chủ đề và 1 chủ đề có nhiều bài đánh giá),
-- đồng thời chống trùng lặp cặp (id_assessment, topic_id) bằng Unique Filtered Index.
-- =====================================================================================
CREATE TABLE dbo.assessment_topic_mapping (
    mapping_id INT NOT NULL,
    id_assessment INT NOT NULL,
    topic_id INT NULL, -- NULL đối với bài thi Exam bị loại khỏi luồng cảnh báo sớm
    mapping_type VARCHAR(30) NOT NULL, -- Primary, Secondary, Excluded_Exam
    mapping_evidence NVARCHAR(1000) NOT NULL,
    evidence_source NVARCHAR(500) NOT NULL,
    confidence VARCHAR(100) NOT NULL,
    mapping_status VARCHAR(20) NOT NULL DEFAULT 'Pending',
    notes NVARCHAR(1000) NULL,

    CONSTRAINT PK_assessment_topic_mapping PRIMARY KEY CLUSTERED (mapping_id),
    CONSTRAINT FK_atm_assessments FOREIGN KEY (id_assessment)
        REFERENCES dbo.assessments (id_assessment),
    CONSTRAINT FK_atm_curriculum_topics FOREIGN KEY (topic_id)
        REFERENCES dbo.curriculum_topics (topic_id),
    CONSTRAINT CK_atm_mapping_type CHECK (mapping_type IN ('Primary', 'Secondary', 'Excluded_Exam')),
    CONSTRAINT CK_atm_mapping_status CHECK (mapping_status IN ('Verified', 'Pending', 'Unmapped'))
);
GO

-- Chống trùng lặp bản ghi ánh xạ giữa cùng 1 bài đánh giá và 1 chủ đề
CREATE UNIQUE NONCLUSTERED INDEX UQ_atm_assessment_topic
ON dbo.assessment_topic_mapping (id_assessment, topic_id)
WHERE topic_id IS NOT NULL;
GO

-- =====================================================================================
-- BẢNG 3: dbo.vle_topic_mapping (Ánh xạ Tài nguyên Học tập VLE -> Chủ đề Kiến thức)
-- =====================================================================================
CREATE TABLE dbo.vle_topic_mapping (
    mapping_id INT NOT NULL,
    id_site INT NOT NULL,
    topic_id INT NULL,
    activity_type VARCHAR(50) NOT NULL,
    mapping_evidence NVARCHAR(1000) NOT NULL,
    evidence_source NVARCHAR(500) NOT NULL,
    mapping_status VARCHAR(20) NOT NULL DEFAULT 'Pending',
    notes NVARCHAR(1000) NULL,

    CONSTRAINT PK_vle_topic_mapping PRIMARY KEY CLUSTERED (mapping_id),
    CONSTRAINT FK_vtm_vle FOREIGN KEY (id_site)
        REFERENCES dbo.vle (id_site),
    CONSTRAINT FK_vtm_curriculum_topics FOREIGN KEY (topic_id)
        REFERENCES dbo.curriculum_topics (topic_id),
    CONSTRAINT CK_vtm_mapping_status CHECK (mapping_status IN ('Verified', 'Pending', 'Unmapped'))
);
GO

-- Chống trùng lặp bản ghi ánh xạ giữa cùng 1 tài nguyên VLE và 1 chủ đề
CREATE UNIQUE NONCLUSTERED INDEX UQ_vtm_site_topic
ON dbo.vle_topic_mapping (id_site, topic_id)
WHERE topic_id IS NOT NULL;
GO

-- =====================================================================================
-- PHẦN TÍCH HỢP LUỒNG XỬ LÝ ĐIỂM YẾU THEO CHỦ ĐỀ (MỤC 5.8 & PHẦN VI)
-- Luồng: studentAssessment -> assessments -> assessment_topic_mapping -> curriculum_topics
-- Nguyên tắc:
-- 1. Loại bỏ hoàn toàn bài thi cuối kỳ (assessment_type <> 'Exam').
-- 2. Giới hạn theo mốc thời gian dự đoán (@CutoffDay, mặc định = 100 cho cảnh báo sớm).
-- 3. Không tự động chia điểm 1 bài đánh giá cho nhiều chủ đề nếu chưa có điểm thành phần:
--    ghi nhận cờ is_multi_topic_assessment để Backend cảnh báo rõ hạn chế độ phân giải.
-- 4. Chỉ đưa vào Prompt Gemini những chủ đề đã có verification_status = 'Verified'
--    (hoặc cho phép chế độ Sandbox kiểm thử khi truyền @AllowPendingForTesting = 1).
-- 5. Tuyệt đối KHÔNG trả về họ tên, email hay id_student sang payload gửi Gemini API.
-- =====================================================================================
CREATE PROCEDURE dbo.usp_GetStudentWeakTopicsForGemini
    @IdStudent INT,
    @CodeModule VARCHAR(10) = 'FFF',
    @CodePresentation VARCHAR(10) = '2013J',
    @CutoffDay INT = 100,
    @WeakScoreThreshold FLOAT = 50.0,
    @MinMappedAssessments INT = 1,
    @AllowPendingForTesting BIT = 0
AS
BEGIN
    SET NOCOUNT ON;

    -- Đếm số lượng chủ đề mà mỗi bài đánh giá đang ánh xạ tới để cảnh báo nếu 1 bài gắn nhiều chủ đề
    WITH AssessmentTopicCounts AS (
        SELECT id_assessment, COUNT(DISTINCT topic_id) AS mapped_topic_count
        FROM dbo.assessment_topic_mapping
        WHERE topic_id IS NOT NULL AND mapping_status <> 'Unmapped'
        GROUP BY id_assessment
    ),
    EligibleAssessments AS (
        SELECT
            a.id_assessment,
            a.assessment_type,
            a.[date] AS due_date,
            CASE WHEN a.[weight] > 0 THEN a.[weight] ELSE 1.0 END AS effective_weight,
            atm.topic_id,
            atm.mapping_type,
            atm.mapping_status,
            ct.topic_code,
            ct.topic_name,
            ct.learning_outcomes,
            ct.verification_status,
            ISNULL(atc.mapped_topic_count, 1) AS mapped_topic_count
        FROM dbo.assessments a
        INNER JOIN dbo.assessment_topic_mapping atm ON a.id_assessment = atm.id_assessment
        INNER JOIN dbo.curriculum_topics ct ON atm.topic_id = ct.topic_id
        LEFT JOIN AssessmentTopicCounts atc ON a.id_assessment = atc.id_assessment
        WHERE a.code_module = @CodeModule
          AND a.code_presentation = @CodePresentation
          AND a.assessment_type <> 'Exam' -- Quy tắc bắt buộc: Không dùng Exam
          AND (a.[date] <= @CutoffDay OR a.assessment_type = 'CMA')
          AND (
              (ct.verification_status = 'Verified' AND atm.mapping_status = 'Verified')
              OR (@AllowPendingForTesting = 1 AND ct.verification_status IN ('Verified', 'Pending'))
          )
    ),
    StudentTopicScores AS (
        SELECT
            ea.topic_id,
            ea.topic_code,
            ea.topic_name,
            ea.learning_outcomes,
            ea.verification_status,
            COUNT(DISTINCT ea.id_assessment) AS total_mapped_assessments,
            COUNT(DISTINCT sa.id_assessment) AS submitted_assessments,
            SUM(ISNULL(sa.score, 0.0) * ea.effective_weight) / NULLIF(SUM(ea.effective_weight), 0) AS topic_weighted_score,
            AVG(ISNULL(sa.score, 0.0)) AS topic_simple_avg_score,
            MAX(CASE WHEN ea.mapped_topic_count > 1 THEN 1 ELSE 0 END) AS has_multi_topic_limitation
        FROM EligibleAssessments ea
        LEFT JOIN dbo.studentAssessment sa
            ON ea.id_assessment = sa.id_assessment
           AND sa.id_student = @IdStudent
           AND sa.date_submitted <= @CutoffDay
        GROUP BY
            ea.topic_id,
            ea.topic_code,
            ea.topic_name,
            ea.learning_outcomes,
            ea.verification_status
    )
    SELECT
        topic_id,
        topic_code,
        topic_name,
        learning_outcomes,
        verification_status,
        total_mapped_assessments,
        submitted_assessments,
        ROUND(topic_weighted_score, 2) AS topic_weighted_score,
        ROUND(topic_simple_avg_score, 2) AS topic_simple_avg_score,
        @WeakScoreThreshold AS configured_weak_threshold,
        CASE
            WHEN has_multi_topic_limitation = 1
            THEN N'Cảnh báo hạn chế dữ liệu: Bài đánh giá bao gồm nhiều chủ đề nhưng OULAD không có điểm từng câu hỏi; điểm chủ đề chỉ mang tính chất ước lượng.'
            ELSE N'Ánh xạ 1-1 ở cấp khối đánh giá.'
        END AS granularity_limitation_note
    FROM StudentTopicScores
    WHERE total_mapped_assessments >= @MinMappedAssessments
      AND topic_weighted_score < @WeakScoreThreshold
    ORDER BY topic_weighted_score ASC;
END;
GO
