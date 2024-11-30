# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import lightgbm as lgb

# Load dataset
#data_path = '/home/joaquintschopp/dataset/competencia_03_ct.csv.gz'
data_path = '~/dataset/competencia_03_ct.csv.gz'
df = pd.read_csv(data_path)

# Transform 'clase_ternaria' into binary target variable
df['target'] = np.where(df['clase_ternaria'] == 'CONTINUA', 0, 1)

# Split data into training, validation, and test sets according to 'foto_mes'
train = df[(df['foto_mes'] >= 201901) & (df['foto_mes'] <= 202106)]
valid = df[df['foto_mes'] == 202107]
test = df[df['foto_mes'] == 202109]

# Define feature columns (excluding non-feature columns)
feature_cols = [col for col in df.columns if col not in ['clase_ternaria', 'target', 'foto_mes']]

X_train = train[feature_cols]
y_train = train['target']

X_valid = valid[feature_cols]
y_valid = valid['target']

X_test = test[feature_cols]

# Placeholder for data processing function
class DataProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass  # Initialize parameters if needed

    def fit(self, X, y=None):
        # Fit processing steps
        return self

    def transform(self, X):
        # Apply processing steps
        return X

# Placeholder for feature creation function
class FeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass  # Initialize parameters if needed

    def fit(self, X, y=None):
        # Fit feature creation steps
        return self

    def transform(self, X):
        # Create new features
        return X

# Hyperparameters for LightGBM
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    # Add other hyperparameters here
}

# Build the pipeline
pipeline = Pipeline([
    ('data_processor', DataProcessor()),
    ('feature_creator', FeatureCreator()),
    # Model is handled separately
])

# Process data
X_train_processed = pipeline.fit_transform(X_train, y_train)
X_valid_processed = pipeline.transform(X_valid)
X_test_processed = pipeline.transform(X_test)

# Prepare datasets for LightGBM
lgb_train = lgb.Dataset(X_train_processed, label=y_train)
lgb_valid = lgb.Dataset(X_valid_processed, label=y_valid)

# Train the LightGBM model
model = lgb.train(
    lgb_params,
    lgb_train,
    valid_sets=[lgb_train, lgb_valid],
    verbose_eval=100,
    # Add early stopping rounds if needed
)

# Predict probabilities on the test set
test_pred_proba = model.predict(X_test_processed)

# Prepare the submission file
submission = pd.DataFrame({
    'numero_de_cliente': test['numero_de_cliente'],
    'Predicted': test_pred_proba
})

# Save to CSV
submission.to_csv('/home/joaquintschopp/salidaspy/submission_py.csv', index=False)
