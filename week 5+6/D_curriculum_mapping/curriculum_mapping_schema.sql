-- =====================================================================================
-- DỰ ÁN: SMART LMS & GENERATIVE AI – PHÂN TÍCH DỮ LIỆU NGƯỜI HỌC VÀ TẠO NỘI DUNG TỰ ĐỘNG
-- TỆP: curriculum_mapping_schema.sql
-- CẬP NHẬT THEO GÓP Ý GIẢNG VIÊN HƯỚNG DẪN:
-- 1. Giữ nguyên 100% các cột dữ liệu gốc của OULAD và bổ sung các cột ánh xạ sang
--    Khối kiến thức Học phần IT1108 – Nhập môn lập trình (15–16 tuần <-> 34 tuần OULAD).
-- 2. Phân biệt rõ trạng thái "Không nộp bài (MISSING_SUBMISSION)" với
--    "Có nộp bài nhưng điểm kém (LOW_SCORE_DESPITE_SUBMISSION)".
-- 3. Tính toán phân vị so với lớp (lower_than_class_pct, ví dụ: "thấp hơn 85% lớp")
--    để phục vụ 3 kịch bản đặt Prompt cho Gemini API (Baseline / Risk-Aware / Detailed Profile).
-- =====================================================================================

USE SmartLMS_DB;
GO

IF OBJECT_ID(N'dbo.usp_GetStudentWeakTopicsForGemini', N'P') IS NOT NULL
    DROP PROCEDURE dbo.usp_GetStudentWeakTopicsForGemini;
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
-- BẢNG 1: dbo.curriculum_topics (Giữ nguyên 11 cột gốc + Thêm 7 cột ánh xạ Khối kiến thức IT1108)
-- =====================================================================================
CREATE TABLE dbo.curriculum_topics (
    -- 11 cột gốc (OULAD Schema Tuần 5 & 6)
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

    -- 7 cột bổ sung ánh xạ sang Khối kiến thức Học phần IT1108 (15-16 tuần)
    it1108_course_code VARCHAR(20) NULL DEFAULT 'IT1108',
    it1108_block_code VARCHAR(50) NULL,
    it1108_block_name NVARCHAR(255) NULL,
    it1108_lessons_mapped NVARCHAR(500) NULL,
    it1108_week_range NVARCHAR(100) NULL,
    it1108_learning_outcomes NVARCHAR(1000) NULL,
    oulad_to_it1108_mapping_rule NVARCHAR(1000) NULL,
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
-- BẢNG 2: dbo.assessment_topic_mapping (Giữ nguyên 9 cột gốc + Thêm 5 cột ánh xạ IT1108)
-- =====================================================================================
CREATE TABLE dbo.assessment_topic_mapping (
    -- 9 cột gốc (OULAD Schema Tuần 5 & 6)
    mapping_id INT NOT NULL,
    id_assessment INT NOT NULL,
    topic_id INT NULL,
    mapping_type VARCHAR(30) NOT NULL,
    mapping_evidence NVARCHAR(1000) NOT NULL,
    evidence_source NVARCHAR(500) NOT NULL,
    confidence VARCHAR(100) NOT NULL,
    mapping_status VARCHAR(20) NOT NULL DEFAULT 'Pending',
    notes NVARCHAR(1000) NULL,

    -- 5 cột bổ sung ánh xạ sang Khối kiến thức & Bài kiểm tra mô phỏng IT1108
    it1108_block_code VARCHAR(50) NULL,
    it1108_mapped_block_name NVARCHAR(255) NULL,
    it1108_simulated_assessment_title NVARCHAR(500) NULL,
    mapping_weight FLOAT NOT NULL DEFAULT 1.0,
    oulad_to_it1108_mapping_rule NVARCHAR(1000) NULL,

    CONSTRAINT PK_assessment_topic_mapping PRIMARY KEY CLUSTERED (mapping_id),
    CONSTRAINT FK_atm_assessments FOREIGN KEY (id_assessment)
        REFERENCES dbo.assessments (id_assessment),
    CONSTRAINT FK_atm_curriculum_topics FOREIGN KEY (topic_id)
        REFERENCES dbo.curriculum_topics (topic_id),
    CONSTRAINT CK_atm_mapping_type CHECK (mapping_type IN ('Primary', 'Secondary', 'Excluded_Exam')),
    CONSTRAINT CK_atm_mapping_status CHECK (mapping_status IN ('Verified', 'Pending', 'Unmapped')),
    CONSTRAINT CK_atm_mapping_weight CHECK (mapping_weight >= 0.0 AND mapping_weight <= 1.0)
);
GO

CREATE UNIQUE NONCLUSTERED INDEX UQ_atm_assessment_topic
ON dbo.assessment_topic_mapping (id_assessment, topic_id)
WHERE topic_id IS NOT NULL;
GO

-- =====================================================================================
-- BẢNG 3: dbo.vle_topic_mapping (Giữ nguyên 8 cột gốc + Thêm 5 cột ánh xạ IT1108)
-- =====================================================================================
CREATE TABLE dbo.vle_topic_mapping (
    -- 8 cột gốc (OULAD Schema Tuần 5 & 6)
    mapping_id INT NOT NULL,
    id_site INT NOT NULL,
    topic_id INT NULL,
    activity_type VARCHAR(50) NOT NULL,
    mapping_evidence NVARCHAR(1000) NOT NULL,
    evidence_source NVARCHAR(500) NOT NULL,
    mapping_status VARCHAR(20) NOT NULL DEFAULT 'Pending',
    notes NVARCHAR(1000) NULL,

    -- 5 cột bổ sung ánh xạ sang Khối kiến thức & Học liệu mô phỏng IT1108
    it1108_block_code VARCHAR(50) NULL,
    it1108_mapped_block_name NVARCHAR(255) NULL,
    it1108_simulated_resource_category NVARCHAR(500) NULL,
    mapping_weight FLOAT NOT NULL DEFAULT 1.0,
    oulad_to_it1108_mapping_rule NVARCHAR(1000) NULL,

    CONSTRAINT PK_vle_topic_mapping PRIMARY KEY CLUSTERED (mapping_id),
    CONSTRAINT FK_vtm_vle FOREIGN KEY (id_site)
        REFERENCES dbo.vle (id_site),
    CONSTRAINT FK_vtm_curriculum_topics FOREIGN KEY (topic_id)
        REFERENCES dbo.curriculum_topics (topic_id),
    CONSTRAINT CK_vtm_mapping_status CHECK (mapping_status IN ('Verified', 'Pending', 'Unmapped')),
    CONSTRAINT CK_vtm_mapping_weight CHECK (mapping_weight >= 0.0 AND mapping_weight <= 1.0)
);
GO

CREATE UNIQUE NONCLUSTERED INDEX UQ_vtm_site_topic
ON dbo.vle_topic_mapping (id_site, topic_id)
WHERE topic_id IS NOT NULL;
GO

-- =====================================================================================
-- STORED PROCEDURE: dbo.usp_GetStudentWeakTopicsForGemini
-- Tích hợp:
-- 1. Ánh xạ từ bài đánh giá OULAD sang Khối kiến thức IT1108 (it1108_block_code, it1108_block_name).
-- 2. Tính phân vị của sinh viên so với toàn lớp ở từng khối kiến thức (lower_than_class_pct, ví dụ: thấp hơn 85% lớp).
-- 3. Phân biệt rõ giữa "Không nộp bài (MISSING_SUBMISSION)" và "Điểm thấp dù đã nộp bài (LOW_SCORE_DESPITE_SUBMISSION)".
-- =====================================================================================
CREATE PROCEDURE dbo.usp_GetStudentWeakTopicsForGemini
    @IdStudent INT,
    @CodeModule VARCHAR(10) = 'FFF',
    @CodePresentation VARCHAR(10) = '2013J',
    @CutoffDay INT = 100,
    @WeakScoreThreshold FLOAT = 50.0,
    @MinMappedAssessments INT = 1,
    @AllowPendingForTesting BIT = 1
AS
BEGIN
    SET NOCOUNT ON;

    WITH EligibleAssessments AS (
        SELECT
            a.id_assessment,
            a.assessment_type,
            a.[date] AS due_date,
            CASE WHEN a.[weight] > 0 THEN a.[weight] ELSE 1.0 END * ISNULL(atm.mapping_weight, 1.0) AS effective_weight,
            ct.topic_id,
            ct.topic_code,
            ct.topic_name,
            ct.it1108_block_code,
            ct.it1108_block_name,
            ct.it1108_lessons_mapped,
            ct.it1108_learning_outcomes,
            ct.verification_status
        FROM dbo.assessments a
        INNER JOIN dbo.assessment_topic_mapping atm ON a.id_assessment = atm.id_assessment
        INNER JOIN dbo.curriculum_topics ct ON atm.topic_id = ct.topic_id
        WHERE a.code_module = @CodeModule
          AND a.code_presentation = @CodePresentation
          AND a.assessment_type <> 'Exam'
          AND (a.[date] <= @CutoffDay OR a.assessment_type = 'CMA')
          AND (
              (ct.verification_status = 'Verified' AND atm.mapping_status = 'Verified')
              OR (@AllowPendingForTesting = 1 AND ct.verification_status IN ('Verified', 'Pending'))
          )
    ),
    AllClassBlockScores AS (
        SELECT
            si.id_student,
            ea.topic_id,
            ea.topic_code,
            ea.it1108_block_code,
            ea.it1108_block_name,
            ea.it1108_lessons_mapped,
            ea.it1108_learning_outcomes,
            COUNT(DISTINCT ea.id_assessment) AS total_mapped_assessments,
            COUNT(DISTINCT sa.id_assessment) AS submitted_assessments,
            AVG(sa.score) AS avg_submitted_score,
            SUM(ISNULL(sa.score, 0.0) * ea.effective_weight) / NULLIF(SUM(ea.effective_weight), 0) AS block_weighted_score
        FROM dbo.studentInfo si
        CROSS JOIN (SELECT DISTINCT topic_id, topic_code, it1108_block_code, it1108_block_name, it1108_lessons_mapped, it1108_learning_outcomes FROM EligibleAssessments) ea_distinct
        INNER JOIN EligibleAssessments ea ON ea.topic_id = ea_distinct.topic_id
        LEFT JOIN dbo.studentAssessment sa
            ON ea.id_assessment = sa.id_assessment
           AND sa.id_student = si.id_student
           AND sa.date_submitted <= @CutoffDay
        WHERE si.code_module = @CodeModule AND si.code_presentation = @CodePresentation
        GROUP BY
            si.id_student,
            ea.topic_id,
            ea.topic_code,
            ea.it1108_block_code,
            ea.it1108_block_name,
            ea.it1108_lessons_mapped,
            ea.it1108_learning_outcomes
    ),
    RankedClassBlockScores AS (
        SELECT
            *,
            ROUND((1.0 - PERCENT_RANK() OVER (PARTITION BY topic_id ORDER BY block_weighted_score ASC)) * 100.0, 1) AS lower_than_class_pct
        FROM AllClassBlockScores
    )
    SELECT
        topic_id,
        topic_code,
        it1108_block_code,
        it1108_block_name,
        it1108_lessons_mapped,
        it1108_learning_outcomes,
        total_mapped_assessments,
        submitted_assessments,
        ROUND(ISNULL(avg_submitted_score, 0.0), 2) AS avg_submitted_score,
        ROUND(block_weighted_score, 2) AS block_weighted_score,
        lower_than_class_pct,
        CASE
            WHEN submitted_assessments = 0
                THEN N'MISSING_SUBMISSION (Chưa nộp bài kiểm tra của khối kiến thức)'
            WHEN submitted_assessments < total_mapped_assessments AND ISNULL(avg_submitted_score, 0.0) < @WeakScoreThreshold
                THEN N'PARTIAL_SUBMISSION_AND_LOW_SCORE (Nộp thiếu bài VÀ điểm bài đã làm thấp)'
            WHEN submitted_assessments < total_mapped_assessments AND ISNULL(avg_submitted_score, 0.0) >= @WeakScoreThreshold
                THEN N'PARTIAL_SUBMISSION_ONLY (Điểm bài đã nộp đạt yêu cầu nhưng còn bỏ sót bài)'
            ELSE N'LOW_SCORE_DESPITE_SUBMISSION (Điểm thấp dù đã nộp đủ bài)'
        END AS submission_vs_score_status
    FROM RankedClassBlockScores
    WHERE id_student = @IdStudent
      AND total_mapped_assessments >= @MinMappedAssessments
      AND (block_weighted_score < @WeakScoreThreshold OR submitted_assessments < total_mapped_assessments)
    ORDER BY block_weighted_score ASC;
END;
GO
