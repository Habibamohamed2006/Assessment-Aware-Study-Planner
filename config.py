"""Central configuration (all generated/assumed constants live here and are documented in data/data_dictionary.md)."""
SEED = 42

# 12 NN input features (raw columns of the processed table). No demographic columns are used.
FEATURES = [
    "clicks_g1", "clicks_g2", "clicks_g3",   # clicks in the 4 weeks before the deadline, by activity group
    "clicks_last_week",                      # clicks in the last week before the deadline (all groups)
    "prev_mean", "prev_last", "n_prev", "has_prev",   # previous scores in the same presentation
    "is_tma", "is_cma", "weight",            # assessment type and weight
    "weeks_since_start",                     # deadline week of the assessment
]
CLICK_COLS = 4          # the first 4 features are click counts (log1p is applied before the NN)
TARGET = "score"

# --- GENERATED (documented rule) -------------------------------------------------
SESSION_CLICKS = 100      # one study session = 100 extra clicks in the 4-week window
SESSION_LAST_WEEK = 0.25  # 25% of the clicks of a session fall in the last week
K_MAX = 8                 # max sessions per assignment considered by the planner
PLAN_HORIZON_WEEKS = 4    # planning starts 4 weeks before the first deadline
SLOTS_PER_WEEK = 3
N_ASSIGN = 3
