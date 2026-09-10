MIMIC-IV-ED Triage Acuity Prediction

Academic machine learning project exploring how far data manipulation and modeling choices can push predictive performance on a real, messy clinical dataset — predicting emergency department (ED) triage acuity (urgency level, 1–5) from vital signs and free-text chief complaints.

This is an independent personal/academic project using the credentialed MIMIC-IV-ED research dataset. No MIMIC data is included in this repository.

Dataset

MIMIC-IV-ED v2.2 — a large, credentialed research dataset of emergency department visits, jointly using the triage and edstays tables. Key fields include vital signs (temperature, heart rate, respiratory rate, oxygen saturation, blood pressure), self-reported pain, free-text chief complaint, and the target label, acuity (1 = most urgent, 5 = least urgent).

Access to MIMIC-IV requires completing PhysioNet's CITI "Data or Specimens Only Research" training and signing a data use agreement.

Why four different methods?

Rather than settling on a single pipeline, this project deliberately explores four progressively different strategies for handling the same core problem — class imbalance, noisy/impossible clinical values, and a highly skewed, high-stakes target (misclassifying a truly urgent patient is far more costly than the reverse). Comparing methods head-to-head surfaces which modeling choices genuinely help versus which just add complexity.

Method 1 — Statistical cleaning + broad model comparison
IQR-based outlier removal on vital signs
TF-IDF vectorization of free-text chiefcomplaint, with feature-count tuned via explained-variance analysis
Compared Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, a Keras neural network, and PyCaret AutoML
Hyperparameter tuning via GridSearchCV; class balancing via SMOTE
Training dynamics experiment: identified "hard" training examples via per-sample log-loss and prediction margin, oversampled them, then used Isolation Forest to filter residual label noise before retraining
Finding: this added complexity did not improve results over the baseline XGBoost model — a useful negative result suggesting the extra noise/complexity hurt generalization more than the hard-example mining helped
Method 2 — ADASYN + correlation-based feature selection
Z-score outlier removal, with acuity-5 rows exempted from removal — acuity 5 was the rarest class in this dataset, so this choice avoided shrinking an already scarce minority class further before later rebalancing
Categorical encoding via factorization, correlation-ranked feature selection
Class balancing via ADASYN across the full dataset before the train/test split, as an exploratory experiment. This setup can introduce data leakage and inflate evaluation performance, so its results are not treated as a valid estimate of generalization.
Random Forest as the evaluation model
Method 3 — Domain-informed rule-based cleaning (exploratory)
Manual clinical binning of vitals into standard categories (e.g., blood pressure staging, tachycardia/bradycardia thresholds) instead of relying purely on statistical cutoffs
Explored domain-informed rules combining chief complaints, vital signs, and acuity labels to identify potentially inconsistent observations — an exploratory experiment rather than a recommended production preprocessing step, given the risk of circular reasoning when rules are derived from the same labels being predicted
Reduced the problem to a binary Urgent vs. Non-urgent framing to directly optimize precision on the clinically critical class, with full precision/recall/ROC evaluation
Method 4 — Production-style pipeline
Consolidated cleaning into a single reproducible pipeline
Persisted fitted label encoders (pickle) and the trained Random Forest model (joblib) for reuse
Validated the saved pipeline against synthetic example patients spanning different acuity levels
Tech stack

Python · pandas · scikit-learn · XGBoost · LightGBM · CatBoost · TensorFlow/Keras · PyCaret · imbalanced-learn (SMOTE, ADASYN) · NLTK · seaborn / matplotlib

Key takeaways
Class imbalance and physiologically impossible values (e.g., systolic ≤ diastolic blood pressure) required deliberate, layered cleaning — no single technique was sufficient on its own
More "sophisticated" doesn't always mean better: the hard-example-mining experiment in Method 1 underperformed the simpler baseline, reinforcing the value of testing assumptions empirically
Combining domain knowledge (clinically meaningful vital-sign thresholds) with statistical methods (Method 3) gave the most interpretable and precision-focused results for the highest-stakes class
Building a reproducible, serializable pipeline (Method 4) is a distinct skill from exploratory modeling — both are demonstrated here
