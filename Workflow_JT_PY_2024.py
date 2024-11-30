# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import lightgbm as lgb
from lightgbm import early_stopping

# Load dataset
print("Loading dataset...")
data_path = '~/buckets/b1/datasets/competencia_03_ct.csv.gz'
df = pd.read_csv(data_path)
print("Dataset loaded.")

# Transform 'clase_ternaria' into binary target variable
print("Transforming 'clase_ternaria' into binary target variable...")
df['target'] = np.where(df['clase_ternaria'] == 'CONTINUA', 0, 1)
print("Transformation complete.")

# Split data into training, validation, and test sets according to 'foto_mes'
print("Splitting data into training, validation, and test sets...")
train = df[(df['foto_mes'] >= 201901) & (df['foto_mes'] <= 202106)]
valid = df[df['foto_mes'] == 202107]
test = df[df['foto_mes'] == 202109]
print("Data splitting complete.")

# Define feature columns (excluding only the target column)
print("Defining feature columns...")
feature_cols = [col for col in df.columns if col not in ['target']]
print("Feature columns defined.")

X_train = train[feature_cols]
y_train = train['target']

X_valid = valid[feature_cols]
y_valid = valid['target']

X_test = test[feature_cols]

print("Preparing training, validation, and test data complete.")

# Placeholder for data processing function
class DataProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass  # Initialize parameters if needed

    def fit(self, X, y=None):
        print("Fitting data processor...")
        # Fit processing steps
        print("Data processor fitted.")
        return self

    def transform(self, X):
        print("Transforming data...")
        # Apply processing steps
        print("Data transformed.")
        return X

# Placeholder for feature creation function
class FeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass  # Initialize parameters if needed

    def fit(self, X, y=None):
        print("Fitting feature creator...")
        # Fit feature creation steps
        print("Feature creator fitted.")
        return self

    def transform(self, X):
        print("Creating new features...")
        # Create new features
        print("New features created.")
        return X

# Hyperparameters for LightGBM
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'first_metric_only': True,
    'boost_from_average': True,
    'feature_pre_filter': False,
    'force_row_wise': True,
    'verbosity': -100,
    'max_depth': -1,
    'min_gain_to_split': 0.0,
    'min_sum_hessian_in_leaf': 0.001,
    'lambda_l1': 0.0,
    'lambda_l2': 0.0,
    'max_bin': 31,
    'bagging_fraction': 1.0,
    'is_unbalance': False,
    'scale_pos_weight': 1.0,
    'learning_rate': 0.02,
    'feature_fraction': 0.01,
    'num_leaves': 2096,
    'min_data_in_leaf': 1000
}


# Build the pipeline
pipeline = Pipeline([
    ('data_processor', DataProcessor()),
    ('feature_creator', FeatureCreator()),
    # Model is handled separately
])

# Process data
print("Processing data...")
X_train_processed = pipeline.fit_transform(X_train, y_train)
X_valid_processed = pipeline.transform(X_valid)
X_test_processed = pipeline.transform(X_test)
print("Data processing complete.")

# Prepare datasets for LightGBM
print("Preparing datasets for LightGBM...")
lgb_train = lgb.Dataset(X_train_processed, label=y_train)
lgb_valid = lgb.Dataset(X_valid_processed, label=y_valid)
print("Datasets prepared.")

# Train the LightGBM model
print("Training LightGBM model...")
model = lgb.train(
    params=lgb_params,
    train_set=lgb_train,
    num_boost_round=9999,
    valid_sets=[lgb_train, lgb_valid],
    callbacks=[early_stopping(stopping_rounds=200)]
)
print("Model training complete.")

# Predict probabilities on the test set
print("Predicting on test set...")
test_pred_proba = model.predict(X_test_processed)
print("Prediction complete.")

# Prepare the submission file
print("Preparing submission file...")
submission = pd.DataFrame({
    'numero_de_cliente': test['numero_de_cliente'],
    'Predicted': test_pred_proba
})
submission.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created.")
