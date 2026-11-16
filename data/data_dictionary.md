# Data Dictionary — Assessment-Aware Study Planner

## 1. Dataset Overview

The project uses the Open University Learning Analytics Dataset (OULAD) to predict student assessment marks and support study planning.

The preprocessing pipeline processes the selected course module (`FFF`) and the following source tables:

- `assessments.csv`
- `studentAssessment.csv`
- `studentVle.csv`
- `vle.csv`

Only Tutor Marked Assessments (TMA) and Computer Marked Assessments (CMA) with known deadlines and non-missing student scores are included. Examinations are excluded.

**Unit of observation:** One student's scored assessment within a course presentation.

The processed dataset contains 20 columns.

Processed file:
`data/processed/oulad_processed.csv`

## 2. Dataset Columns

| Column | Type | Source / Status | Description |
|---|---|---|---|
| `code_module` | String | OULAD | Course module code. The selected module is FFF. |
| `code_presentation` | String | OULAD | Course presentation identifier. |
| `id_student` | Integer | OULAD | Student identifier. |
| `id_assessment` | Integer | OULAD | Assessment identifier. |
| `assessment_type` | String | OULAD | Assessment type: TMA or CMA. |
| `date` | Numeric | OULAD | Assessment deadline in days from the course start. |
| `week_deadline` | Integer | Derived | Deadline week, calculated as floor(date / 7). |
| `clicks_g1` | Numeric | Derived from OULAD | Total clicks on content and reading-related activities during the four weeks before the deadline. |
| `clicks_g2` | Numeric | Derived from OULAD | Total clicks on interaction and collaboration activities during the four weeks before the deadline. |
| `clicks_g3` | Numeric | Derived from OULAD | Total clicks on all remaining activity types during the four weeks before the deadline. |
| `clicks_last_week` | Numeric | Derived from OULAD | Total clicks across all activity groups during the week immediately before the deadline. |
| `prev_mean` | Numeric | Derived | Mean score of the student's previous scored assessments in the same presentation. Defaults to 50.0 if no previous score exists. |
| `prev_last` | Numeric | Derived | Score of the student's immediately previous scored assessment in the same presentation. Defaults to 50.0 if no previous score exists. |
| `n_prev` | Integer | Derived | Number of previous scored assessments for the student in the same presentation. |
| `has_prev` | Binary | Derived | 1 if the student has previous scored assessments; otherwise 0. |
| `is_tma` | Binary | Derived | 1 if the assessment type is TMA; otherwise 0. |
| `is_cma` | Binary | Derived | 1 if the assessment type is CMA; otherwise 0. |
| `weight` | Numeric | OULAD | Assessment weight. |
| `weeks_since_start` | Integer | Derived | Equal to `week_deadline` in the preprocessing pipeline. |
| `score` | Numeric | OULAD | Student's assessment mark. This is the prediction target. |

## 3. Click Activity Grouping

VLE activity types are divided into three groups by the preprocessing function:

**Group 1 — Content and Reading Activities (`clicks_g1`)**

Includes:
- resource
- oucontent
- url
- page
- subpage
- glossary
- folder
- htmlactivity
- dataplus

**Group 2 — Interaction and Collaboration Activities (`clicks_g2`)**

Includes:
- forumng
- oucollaborate
- ouwiki
- ouelluminate
- questionnaire
- dualpane

**Group 3 — Other Activities (`clicks_g3`)**

Any activity type not included in Group 1 or Group 2 is assigned to Group 3.

## 4. Click Aggregation and Missing Values

For each assessment, click activity is aggregated over the four weeks preceding its deadline.

The aggregation window is:

- Starting week: `week_deadline - 4`
- Ending week: `week_deadline - 1`
- The deadline week itself is excluded.

`clicks_last_week` represents the total clicks across all activity groups during `week_deadline - 1`.

**Important:** `clicks_last_week` is a subset of the activity already included in the four-week click aggregates. It is included as a separate feature to represent recent activity intensity.

Missing click aggregates are filled with zero.

## 5. Previous Assessment Features

Previous assessment information is calculated separately for each student within each course presentation.

- `prev_mean`: Mean of all earlier scored assessments.
- `prev_last`: Score of the immediately preceding scored assessment.
- `n_prev`: Number of earlier scored assessments.
- `has_prev`: Indicates whether at least one previous scored assessment exists.

When no previous assessment is available, `prev_mean` and `prev_last` are set to 50.0. The `has_prev` feature distinguishes this default value from actual previous-score information.

## 6. Neural Network Input Features

The Neural Network uses 12 input features selected from the processed dataset.

| Feature | Description |
|---|---|
| `clicks_g1` | Content and reading clicks |
| `clicks_g2` | Interaction and collaboration clicks |
| `clicks_g3` | Other activity clicks |
| `clicks_last_week` | Clicks in the last week before the deadline |
| `prev_mean` | Mean of previous assessment scores |
| `prev_last` | Most recent previous assessment score |
| `n_prev` | Number of previous scored assessments |
| `has_prev` | Whether previous scored assessments exist |
| `is_tma` | TMA indicator |
| `is_cma` | CMA indicator |
| `weight` | Assessment weight |
| `weeks_since_start` | Deadline week |

**Target variable:** `score`

No demographic attributes are included in the Neural Network input features.

## 7. Neural Network Input Transformation

Before standardization, the first four click-count features are transformed using the natural logarithm transformation `log1p`.

After this transformation, all input features are standardized using the training data mean and standard deviation.

The same training statistics are used when transforming validation and prediction inputs.

## 8. Neural Network Architecture and Training

The project uses a NumPy-based Multi-Layer Perceptron (MLP).

**Architecture:**
- Input layer: 12 features.
- First hidden layer: 16 neurons.
- Second hidden layer: 8 neurons.
- Output layer: 1 predicted score.
- Hidden-layer activation: `tanh`.
- Output activation: Linear.

**Training configuration in the model implementation:**

| Parameter | Value |
|---|---:|
| Learning rate | 0.003 |
| Batch size | 128 |
| Maximum epochs | 300 |
| Early stopping patience | 25 |
| L2 regularization | 0.0001 |
| Loss function | Mean Squared Error (MSE) |
| Optimizer | Adam |

The target scores are divided by 100 during training. Predictions are converted back to the 0–100 scale and clipped to that range.

Early stopping monitors validation loss and restores the best-performing model parameters.

## 9. Generated Planning Assumptions

The planner uses hypothetical study sessions to estimate how additional effort may affect predicted assessment marks.

These values are configurable assumptions, not observed OULAD measurements.

| Parameter | Value | Description |
|---|---:|---|
| `SEED` | 42 | Central random seed configuration. |
| `SESSION_CLICKS` | 100 | Each hypothetical session adds 100 clicks. |
| `SESSION_LAST_WEEK` | 0.25 | 25% of added clicks are assigned to the last-week click feature. |
| `K_MAX` | 8 | Maximum number of additional sessions considered per assignment. |
| `PLAN_HORIZON_WEEKS` | 4 | Configured planning start offset before the first deadline. |
| `SLOTS_PER_WEEK` | 3 | Default weekly scheduling capacity. |
| `N_ASSIGN` | 3 | Number of assignments handled by the planner. |

### 9.1 Hypothetical Session Generation

The `add_sessions` function creates a modified copy of a student's click features for what-if prediction.

For `k` additional sessions:

1. Total added clicks equal `k × SESSION_CLICKS`.
2. Added clicks are distributed across `clicks_g1`, `clicks_g2`, and `clicks_g3` according to the student's existing activity proportions.
3. If the student has no clicks in the three groups, the added clicks are distributed equally across the groups.
4. `clicks_last_week` is increased by `SESSION_LAST_WEEK` multiplied by the total added clicks.

These additional clicks are hypothetical planning inputs. They are not actual observed student activity.

### 9.2 Prediction Across Session Counts

The NN predictor generates a predicted mark for each assignment for session counts from zero to `K_MAX`.

When monotonicity enforcement is enabled, cumulative maximum is applied to the prediction sequence. This ensures that the adjusted predicted mark does not decrease as the number of hypothetical sessions increases.

### 9.3 Average-Mark Baseline

The `AvgPredictor` is used as an NN-off baseline.

It predicts using the average-mark baseline and ignores the effect of additional sessions. Therefore, it returns the same predicted mark for every session count.

## 10. Data Privacy and Scope

The Neural Network uses assessment performance, previous assessment history, assessment information, and VLE activity features.

No demographic attributes are included in the 12 NN input features.

Student and assessment identifiers are retained in the processed dataset to associate records with students and assessments. They are not part of the configured NN input feature list.

## 11. Reproducibility and Implementation Files

The main implementation files relevant to the dataset and prediction pipeline are:

- `config.py`: Central configuration, feature list, target, and generated planning constants.
- `data/preprocessing.py`: Loads the OULAD source tables, groups VLE activity, aggregates clicks, and creates the processed dataset.
- `nn/predict.py`: Generates hypothetical session inputs and predicted marks.
- `nn/model.py`: Defines the MLP architecture, input transformation, training, prediction, saving, and loading.

The configured random seed is 42. The exact use of this seed throughout dataset generation and training depends on the corresponding execution scripts.