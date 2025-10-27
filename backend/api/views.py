# backend/api/views.py
import io
import base64
import matplotlib
matplotlib.use('Agg')  # backend sin GUI, evita el RuntimeError de Tkinter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def train_rf(request):
    try:
        # Parámetros del usuario
        percentage = int(request.GET.get('percentage', 70))
        tree_index = int(request.GET.get('tree_index', 1)) - 1

        # Cargar dataset
        data = load_iris()
        X = data.data
        y = data.target

        # Dividir datos (mínimo 20% test, máximo 80% train)
        train_size = max(min(percentage / 100, 0.8), 0.6)
        test_size = 1 - train_size

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Entrenar Random Forest
        rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            min_samples_leaf=3,
            max_features='sqrt',
            random_state=42
        )
        rf.fit(X_train, y_train)

        # Validar índice del árbol
        tree_index = max(0, min(tree_index, len(rf.estimators_)-1))

        # Predicción y evaluación
        y_pred = rf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        report = classification_report(y_test, y_pred, output_dict=True)

        # --- Gráfico del límite de decisión ---
        X_plot = X_train[:, :2]  # solo primeras dos features para plot
        xx, yy = np.meshgrid(
            np.linspace(X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1, 100),
            np.linspace(X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1, 100)
        )

        # Creamos grid completo con 4 features (otras dos fijas en promedio)
        avg_rest = np.mean(X_train[:, 2:], axis=0)
        grid_full = np.c_[xx.ravel(), yy.ravel(), np.full_like(xx.ravel(), avg_rest[0]), np.full_like(xx.ravel(), avg_rest[1])]
        Z = rf.estimators_[tree_index].predict(grid_full)
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(6, 6))
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_train, s=20, edgecolor='k')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Random Forest - Límite de decisión')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        decision_plot = base64.b64encode(buf.getvalue()).decode('utf-8')

        # --- Árbol seleccionado ---
        plt.figure(figsize=(20, 10))
        plot_tree(
            rf.estimators_[tree_index],
            feature_names=data.feature_names,
            class_names=data.target_names,
            filled=True
        )
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        tree_plot = base64.b64encode(buf.getvalue()).decode('utf-8')

        return JsonResponse({
            'f1_score': round(float(f1), 4),
            'report': report,
            'decision_plot': decision_plot,
            'tree_plot': tree_plot,
            'tree_index': tree_index
        })

    except Exception as e:
        return JsonResponse({'error': str(e)})
