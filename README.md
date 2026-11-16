# Assessment-Aware Study Planner

## Project Overview

This project is an AI-based study planner that helps students organize their study sessions and plan for upcoming assessments.

It combines three AI techniques:
- Neural Network to predict assessment marks.
- Search algorithms to find a suitable study plan.
- CSP to schedule study sessions while respecting deadlines and constraints.

The project also includes a Streamlit GUI to make it easier to use and test the different components.

## Dataset

We used the Open University Learning Analytics Dataset (OULAD).

The project focuses on:
- Module: FFF
- Presentations: 2013B, 2013J, 2014B, 2014J

The processed dataset is located in:

`data/processed/oulad_processed.csv`

For more details about the dataset columns, see:

`data/data_dictionary.md`

## Project Structure

```text
study_planner/
│
├── app.py
├── config.py
├── run_demo.py
├── README.md
├── requirements.txt
├── pytest.ini
│
├── data/
├── nn/
├── search/
├── csp/
├── integration/
├── eval/
├── gui/
├── models/
├── results/
└── tests/
```

## Main Components

### 1. Neural Network

Predicts student assessment marks using activity data and previous performance.

The trained model is saved in:

`models/study_planner_mlp.npz`

A baseline model is also included for comparison.

### 2. Search

The project implements:
- Uniform Cost Search (UCS)
- Greedy Best-First Search
- A* Search

The algorithms are compared using search cost and the number of expanded and generated nodes.

### 3. CSP

The project includes three CSP techniques:
- Backtracking
- Forward Checking
- Arc Consistency (AC-3)

They are used to schedule study sessions while satisfying the defined constraints.

### 4. Integration

The integration module connects the NN, Search, and CSP components to generate a study plan.

## Installation

The project can be run using Visual Studio Code on Windows.

**1. Create a virtual environment**

```powershell
python -m venv .venv
```

**2. Activate it**

```powershell
.venv\Scripts\Activate.ps1
```

If activation is blocked, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

**3. Install the required packages**

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the Project

**Run the GUI:**

```powershell
python -m streamlit run app.py
```

The application will open in the browser at:

`http://localhost:8501`

**Run the demo:**

```powershell
python run_demo.py
```

## Training the Model

To train the neural network again:

```powershell
python -m nn.train
```

A trained model is already included in the `models` folder, so training is not required before running the application.

## Experiments and Results

The `results/` folder contains the experiment results, CSV files, summaries, and plots.

The experiments cover:
- Search algorithm comparison.
- CSP algorithm comparison.
- Neural Network versus the average-mark baseline.
- Prediction monotonicity.

## Tests

To run all tests:

```powershell
python -m pytest -q
```

To see detailed test output:

```powershell
python -m pytest -v
```

Tests cover normal cases, edge cases, impossible cases, and the different project components.

## Limitations

- Predicted marks are estimates and are not guaranteed actual results.
- Student activity and assessment performance may be related, but this does not prove causation.
- The planner depends on the dataset and the scheduling assumptions used in the project.

## Project Information

**Project:** Assessment-Aware Study Planner

**Course Project:** P1 — Search + CSP + Neural Network

**Dataset:** Open University Learning Analytics Dataset (OULAD)