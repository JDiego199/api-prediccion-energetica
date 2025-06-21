from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

# Cargar el modelo entrenado y las columnas de features
MODEL_PATH = 'model/energy_predictor_model.joblib'
model_data = joblib.load(MODEL_PATH)
model = model_data['model']
feature_cols = model_data['feature_cols']

# Cargar el dataset histórico para calcular lags y rolling
DATASET_PATH = 'data/df_dataset_unidos5.csv'  # Cambia la ruta si es necesario
df_hist = pd.read_csv(DATASET_PATH)

def add_lags_and_rolling(df, empresa, año, mes):
    """
    Calcula los lags y rolling para una empresa, año y mes dados,
    usando el dataset histórico.
    Devuelve un diccionario con los valores calculados.
    """
    # Filtrar datos históricos de la empresa
    df_empresa = df[df['IdEmpresa'] == empresa].copy()
    # Ordenar por año y mes
    df_empresa = df_empresa.sort_values(['Año', 'IdMes'])
    # Crear una fila vacía para el mes a predecir
    nueva_fila = {
        'IdEmpresa': empresa,
        'Año': año,
        'IdMes': mes
    }
    # Concatenar la nueva fila al final
    df_empresa = pd.concat([df_empresa, pd.DataFrame([nueva_fila])], ignore_index=True)
    # Recalcular índices
    df_empresa = df_empresa.reset_index(drop=True)

    # Calcular lags
    lag_periods = [1, 2, 3, 6, 12]
    for lag in lag_periods:
        df_empresa[f'Energía Facturada (MWh)_lag_{lag}'] = df_empresa['Energía Facturada (MWh)'].shift(lag)
    # Calcular rolling
    rolling_windows = [3, 6, 12]
    for window in rolling_windows:
        df_empresa[f'Energía Facturada (MWh)_rolling_mean_{window}'] = df_empresa['Energía Facturada (MWh)'].rolling(window=window, min_periods=1).mean()
        df_empresa[f'Energía Facturada (MWh)_rolling_std_{window}'] = df_empresa['Energía Facturada (MWh)'].rolling(window=window, min_periods=1).std()
        df_empresa[f'Energía Facturada (MWh)_rolling_trend_{window}'] = df_empresa['Energía Facturada (MWh)'].rolling(window=window, min_periods=2).apply(
            lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0
        )
    # Tomar la última fila (la del mes a predecir)
    ultima = df_empresa.iloc[-1]
    # Extraer solo los lags y rolling
    lags_rolling = {col: ultima[col] for col in df_empresa.columns if 'lag_' in col or 'rolling_' in col}
    return lags_rolling

# --- Clase para ingeniería de features mínima para predicción ---
class MinimalFeatureEngineer:
    def __init__(self, feature_cols):
        self.feature_cols = feature_cols

    def create_seasonal_features(self, df):
        df['mes_sin'] = np.sin(2 * np.pi * df['IdMes'] / 12)
        df['mes_cos'] = np.cos(2 * np.pi * df['IdMes'] / 12)
        df['trimestre_sin'] = np.sin(2 * np.pi * ((df['IdMes'] - 1) // 3 + 1) / 4)
        df['trimestre_cos'] = np.cos(2 * np.pi * ((df['IdMes'] - 1) // 3 + 1) / 4)
        df['semestre_sin'] = np.sin(2 * np.pi * ((df['IdMes'] - 1) // 6 + 1) / 2)
        df['semestre_cos'] = np.cos(2 * np.pi * ((df['IdMes'] - 1) // 6 + 1) / 2)
        return df

    def add_temporal_features(self, df):
        df['año_mes'] = df['Año'] * 100 + df['IdMes']
        df['tendencia_temporal'] = (df['Año'] - df['Año'].min()) * 12 + df['IdMes']
        return df

    def transform(self, df):
        df = self.create_seasonal_features(df)
        df = self.add_temporal_features(df)
        # Rellenar features faltantes con 0 (lags y rolling no disponibles en predicción individual)
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
        return df[self.feature_cols]

# Instanciar el feature engineer
feature_engineer = MinimalFeatureEngineer(feature_cols)

# --- Flask API ---
app = Flask(__name__)
CORS(app, origins=["*"])

@app.route('/predict', methods=['POST'])
def predict():
    """
    Espera un JSON con los campos:
    IdEmpresa, Año, IdMes, temperatura, precipitacion, PIB_mensual_interpolado, COSTO_CANASTA, INGRESO_FAMILIAR_MENSUAL
    """
    data = request.get_json()
    # Validar campos requeridos
    required_fields = [
        'IdEmpresa', 'Año', 'IdMes', 'temperatura', 'precipitacion',
        'PIB_mensual_interpolado', 'COSTO_CANASTA', 'INGRESO_FAMILIAR_MENSUAL'
    ]
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Campo requerido faltante: {field}'}), 400

    # Calcular lags y rolling usando el dataset histórico
    lags_rolling = add_lags_and_rolling(
        df_hist,
        data['IdEmpresa'],
        data['Año'],
        data['IdMes']
    )
    # Unir los datos recibidos con los lags y rolling
    input_data = {**data, **lags_rolling}
    input_df = pd.DataFrame([input_data])
    # Ingeniería de features
    X = feature_engineer.transform(input_df)
    # Predecir
    pred = model.predict(X)[0]
    return jsonify({'status': 'success', 'prediccion_MWh': float(pred)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)