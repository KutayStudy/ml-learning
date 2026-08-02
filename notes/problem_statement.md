## Problem statement

Target is p_recall (0-1, probability of recall). Regression problem. One row = one user practicing one word at a given time.

Sampled 2,500 users / 16,382 sessions from the Duolingo learning traces dataset, using user-level random sampling so each user's full session history stays together (needed GroupKFold later, not a plain random split).

Picked this over the other candidates because it needs grouped splitting (real scenario, not a toy one), it maps onto Duolingo's own published half-life regression work, and there's real feature engineering room (history_seen/history_correct, lag_days, word difficulty join).

Leakage risk: p_recall = session_correct / session_seen exactly. Don't use those two columns as features, ever.