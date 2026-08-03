# Data Dictionary — Duolingo Flagship Dataset

Source: `data/duolingo_flagship.csv` (2,500 users, 16,382 sessions, user-level sample of Duolingo Learning Traces)

| Column | Type | Meaning | % Missing | Cardinality | Notes |
|---|---|---|---|---|---|
| `practice_time` | datetime | Timestamp of the practice session | 0% | 11,275 | |
| `user_id` | str | Learner identifier | 0% | 2,500 | Not a feature — split/group key only |
| `ui_language` | str | Language the app UI is shown in | 0% | 4 | Usable categorical |
| `learning_language` | str | Language being learned | 0% | 6 | Usable categorical |
| `surface_form` | str | Exact word form shown to the user | 0% | 2,841 | Too high-cardinality for direct use |
| `lemma` | str | Dictionary/base form of the word | 0% | 2,110 | Too high-cardinality for direct use |
| `pos` | str | Part of speech | 0% | 47 | Usable categorical |
| `grammar_tags` | str | Grammatical tags (tense, person, number, etc.) | 10.46% | 323 | Real missing, not random — check against `pos` in EDA (Day 8-9) |
| `lag_days` | float | Days since this lexeme was last practiced | 0% | 6,301 | Usable numeric |
| `history_seen` | int | Times this lexeme seen before this session | 0% | 265 | Usable numeric |
| `history_correct` | int | Times answered correctly before this session | 0% | 247 | Usable numeric |
| `session_seen` | int | Times seen in *this* session | 0% | 17 | **FORBIDDEN — target component, never a feature** |
| `session_correct` | int | Times correct in *this* session | 0% | 17 | **FORBIDDEN — target component, never a feature** |
| `p_recall` | float | **TARGET.** Probability of recall this session (`session_correct/session_seen`) | 0% | 29 | Median = 1.0, heavily clumped; effectively discrete despite being float |
| `lexeme_id` | str | Unique word identifier | 0% | 3,315 | Too high-cardinality for direct use — valid JOIN key to `word_difficulty.csv` |

## Notes

Target is skewed toward 1.0, so a dummy baseline predicting the median will look deceptively good on Day 12. Same trap as majority-class accuracy on imbalanced classification, just the regression version.

p_recall only takes 29 distinct values since it's session_correct/session_seen with session_seen capped around 17.it's a ratio of small integers, not truly continuous. Worth remembering when residuals look lumpy later.

grammar_tags missingness might track pos (some word types just don't have grammar tags) rather than being random — check this in EDA before deciding how to fill it.

surface_form, lemma, and lexeme_id are too high-cardinality to one-hot encode. lexeme_id is still useful as a join key into word_difficulty.csv.