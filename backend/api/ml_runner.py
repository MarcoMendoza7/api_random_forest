import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestRegressor
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from ml.plots import plot_decision_boundary, plot_tree_image


DATA_PATH = "/home/marco/Documentos/ZZZ/datasets/TotalFeatures-ISCXFlowMeter.csv"

def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return train_set, val_set, test_set

def remove_labels(df, label_name):
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return X, y

def train_and_evaluate(params):
    """
    params: dict con keys
        - num_samples: int
        - tree_index: int (del bosque)
    """
    num_samples = params.get("num_samples")
    tree_index = params.get("tree_index", 0)

    # 1. Cargar dataset
    df = pd.read_csv(DATA_PATH)

    # 2. Seleccionar el número de filas que quiere el usuario
    if num_samples:
        df = df.sample(n=num_samples, random_state=42)

    # 3. Transformar variable de salida a numérica para correlaciones
    X = df.copy()
    X['calss'] = X['calss'].factorize()[0]

    # 4. División del dataset
    train_set, val_set, test_set = train_val_test_split(X, stratify='calss')
    X_train, y_train = remove_labels(train_set, 'calss')
    X_val, y_val = remove_labels(val_set, 'calss')

    # 5. Escalado
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)

    # 6. Modelos DecisionTree
    clf_tree = DecisionTreeClassifier(random_state=42)
    clf_tree.fit(X_train, y_train)
    clf_tree_scaled = DecisionTreeClassifier(random_state=42)
    clf_tree_scaled.fit(X_train_scaled, y_train)

    # 7. Predicciones
    y_train_pred = clf_tree.predict(X_train)
    y_train_scaled_pred = clf_tree_scaled.predict(X_train_scaled)
    y_val_pred = clf_tree.predict(X_val)
    y_val_scaled_pred = clf_tree_scaled.predict(X_val_scaled)

    # 8. Calcular F1 score
    f1_train = {
        "without_scaling": f1_score(y_train, y_train_pred, average='weighted'),
        "with_scaling": f1_score(y_train, y_train_scaled_pred, average='weighted')
    }
    f1_val = {
        "without_scaling": f1_score(y_val, y_val_pred, average='weighted'),
        "with_scaling": f1_score(y_val, y_val_scaled_pred, average='weighted')
    }

    # 9. Random Forest Regressor (usando solo 2 features para gráfico)
    X_train_reduced = X_train[['min_flowpktl', 'flow_fin']]
    reg_tree = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    reg_tree.fit(X_train_reduced, y_train)

    # 10. Generar gráficos
    # Límite de decisión
    decision_plot_b64 = plot_decision_boundary(reg_tree, X_train_reduced.values, y_train)

    # Árbol específico
    tree_plot_b64 = plot_tree_image(reg_tree, tree_index, X_train_reduced.columns, ['benign','adware','malware'])

    return {
        "f1_train": f1_train,
        "f1_val": f1_val,
        "decision_boundary": decision_plot_b64,
        "tree_image": tree_plot_b64
    }
