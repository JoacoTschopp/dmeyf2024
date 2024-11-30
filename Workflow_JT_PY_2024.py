# Importar las librerías necesarias
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from lightgbm import LGBMClassifier
import optuna
from sklearn.metrics import make_scorer

# Cargar el conjunto de datos
print("Cargando el conjunto de datos...")
data_path = '~/buckets/b1/datasets/competencia_03_ct.csv.gz'
df = pd.read_csv(data_path)
print("Conjunto de datos cargado.")

# Transformar 'clase_ternaria' en variable objetivo binaria
print("Transformando 'clase_ternaria' en variable objetivo binaria...")
df['target'] = np.where(df['clase_ternaria'] == 'CONTINUA', 0, 1)
print("Transformación completa.")

# Dividir los datos en entrenamiento, validación y prueba según 'foto_mes'
print("Dividiendo los datos en entrenamiento, validación y prueba...")
train = df[(df['foto_mes'] >= 201901) & (df['foto_mes'] <= 202106)]
valid = df[df['foto_mes'] == 202107]
test = df[df['foto_mes'] == 202109]
print("División de datos completa.")

# Definir las columnas de características (excluyendo 'target' y 'clase_ternaria')
print("Definiendo columnas de características...")
feature_cols = [col for col in df.columns if col not in ['target', 'clase_ternaria']]
print("Columnas de características definidas.")

X_train = train[feature_cols]
y_train = train['target']

X_valid = valid[feature_cols]
y_valid = valid['target']

X_test = test[feature_cols]

print("Preparación de datos de entrenamiento, validación y prueba completa.")

# Definir el transformador de procesamiento de datos
class DataProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        print("Ajustando el procesador de datos...")
        # Implementa los pasos de ajuste si es necesario
        print("Procesador de datos ajustado.")
        return self

    def transform(self, X):
        print("Transformando datos...")
        # Aplica los pasos de transformación
        print("Datos transformados.")
        return X

# Definir el transformador de creación de características
class FeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        print("Ajustando el creador de características...")
        # Implementa los pasos de ajuste si es necesario
        print("Creador de características ajustado.")
        return self

    def transform(self, X):
        print("Creando nuevas características...")
        # Crea nuevas características
        print("Nuevas características creadas.")
        return X

# Construir el Pipeline para el procesamiento de datos y creación de características
pipeline = Pipeline([
    ('data_processor', DataProcessor()),
    ('feature_creator', FeatureCreator())
])

# Procesar los datos
print("Procesando los datos...")
X_train_processed = pipeline.fit_transform(X_train, y_train)
X_valid_processed = pipeline.transform(X_valid)
X_test_processed = pipeline.transform(X_test)
print("Datos procesados.")

# Definir la función de ganancia personalizada
def custom_gain(y_true, y_pred_proba, threshold=0.025):
    y_pred = (y_pred_proba >= threshold).astype(int)
    gain = 0
    for true, pred in zip(y_true, y_pred):
        if pred == 1:
            if true == 1:
                gain += 273000  # Predicción correcta de BAJA+2
            else:
                gain -= 7000    # Predicción incorrecta de BAJA+2
    return gain

# Definir la función objetivo para Optuna
def objective(trial):
    # Especificar los hiperparámetros a optimizar
    param = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'None',  # Desactivamos las métricas internas de LightGBM
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
        'extra_trees': True,
        'n_estimators': 9999,  # Equivalente a 'num_iterations'
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.3),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.01, 0.9),
        'num_leaves': trial.suggest_int('num_leaves', 8, 4096),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50000)
    }

    # Crear el modelo
    model = LGBMClassifier(**param)
    
    # Entrenar el modelo
    model.fit(
        X_train_processed, y_train,
        eval_set=[(X_valid_processed, y_valid)],
        eval_metric='binary_logloss',
        early_stopping_rounds=200,
        verbose=False
    )
    
    # Obtener las predicciones en el conjunto de validación
    y_pred_proba = model.predict_proba(X_valid_processed)[:, 1]
    
    # Calcular la ganancia personalizada
    gain = custom_gain(y_valid.values, y_pred_proba)
    
    # Optuna maximiza la función objetivo
    return gain

# Crear el estudio de Optuna
study = optuna.create_study(direction='maximize')
print("Iniciando la optimización de hiperparámetros con Optuna...")
study.optimize(objective, n_trials=50)
print("Optimización completa.")

# Obtener los mejores hiperparámetros
best_params = study.best_params
print("Mejores hiperparámetros:", best_params)

# Actualizar los parámetros con los valores fijos
best_params.update({
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'None',
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
    'extra_trees': True,
    'n_estimators': 9999
})

# Entrenar el modelo final con los mejores hiperparámetros
print("Entrenando el modelo final con los mejores hiperparámetros...")
final_model = LGBMClassifier(**best_params)

final_model.fit(
    X_train_processed, y_train,
    eval_set=[(X_valid_processed, y_valid)],
    eval_metric='binary_logloss',
    early_stopping_rounds=200,
    verbose=False
)
print("Entrenamiento del modelo final completo.")

# Predecir en el conjunto de prueba
print("Prediciendo en el conjunto de prueba...")
test_pred_proba = final_model.predict_proba(X_test_processed)[:, 1]
print("Predicción completa.")

# Preparar el archivo de salida
print("Preparando el archivo de salida...")
submission = pd.DataFrame({
    'numero_de_cliente': test['numero_de_cliente'],
    'Predicted': test_pred_proba
})
submission.to_csv('submission.csv', index=False)
print("Archivo de salida 'submission.csv' creado.")
