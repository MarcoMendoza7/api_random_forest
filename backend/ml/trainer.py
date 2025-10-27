import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def train_model(dataset_path, train_size, selected_tree):
    # 1. Leer dataset
    df = pd.read_csv(dataset_path)

    # 2. Separar variables
    X = df.drop('target', axis=1)
    y = df['target']

    # 3. División personalizada
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_size, random_state=42)

    # 4. Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 5. Entrenar bosque sin escalar
    rf_unscaled = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_unscaled.fit(X_train, y_train)

    # 6. Entrenar bosque escalado
    rf_scaled = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_scaled.fit(X_train_scaled, y_train)

    # 7. Predicciones
    y_pred_unscaled = rf_unscaled.predict(X_val)
    y_pred_scaled = rf_scaled.predict(X_val_scaled)

    # 8. Evaluaciones
    f1_unscaled = f1_score(y_val, y_pred_unscaled, average='weighted')
    f1_scaled = f1_score(y_val, y_pred_scaled, average='weighted')

    # 9. Extraer árbol específico
    selected_estimator = rf_scaled.estimators_[selected_tree]

    return {
        'f1_scaled': f1_scaled,
        'f1_unscaled': f1_unscaled,
        'comparison': 'Scaled mejor' if f1_scaled > f1_unscaled else 'Unscaled mejor',
        'tree': selected_estimator
    }

