import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import difflib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class APS_Solver:

    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = None
        self.preprocessor_cluster = None

    # -----------------------------
    # LIMPIEZA GENERAL Y FEATURES
    # -----------------------------
    def _preprocess(self, df, fit=True):
        df = df.copy().drop_duplicates()
    
        revenue_col = df["Revenue"] if "Revenue" in df.columns else None
        if revenue_col is not None:
            df = df.drop(columns=["Revenue"])
    
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()
    
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())
    
        for col in cat_cols:
            df[col] = df[col].astype(str).str.strip().str.lower()
            valid_vals = df[col].dropna().unique()
            df[col] = df[col].apply(lambda x: difflib.get_close_matches(x, valid_vals, n=1, cutoff=0.7)[0] 
                                    if difflib.get_close_matches(x, valid_vals, n=1, cutoff=0.7) else x)
    
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
        if fit:
            self.scaler = StandardScaler()
            df[num_cols] = self.scaler.fit_transform(df[num_cols])
            self.feature_columns = df.columns
        else:
            df[num_cols] = self.scaler.transform(df[num_cols])
            df = df.reindex(columns=self.feature_columns, fill_value=0)
    
        if revenue_col is not None:
            df["Revenue"] = revenue_col.astype(int)
    
        if all(x in df.columns for x in ["ProductRelated_Duration", "Informational_Duration", "Administrative_Duration"]):
            df["Duration_Total"] = (df["ProductRelated_Duration"] + 
                                    df["Informational_Duration"] + 
                                    df["Administrative_Duration"])
            df = df.drop(columns=["ProductRelated_Duration", "Informational_Duration", "Administrative_Duration"])
    
        return df

    # -----------------------------
    # ENTRENAMIENTO
    # -----------------------------
    def train_model(self, file_path):
        self.cluster_data(file_path, save_csv=True)
        df = pd.read_csv("database_clusters.csv")
        df = self._preprocess(df, fit=True)

        X = df.drop(columns=["Revenue"])
        y = df["Revenue"]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        self.model = RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=42, class_weight="balanced"
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_val)
        print("==== Métricas de validación ====")
        print("Tasa de error:", 1 - accuracy_score(y_val, y_pred))
        print("Precisión:", precision_score(y_val, y_pred))
        print("Recall:", recall_score(y_val, y_pred))
        print("F1-score:", f1_score(y_val, y_pred))

    # -----------------------------
    # TESTEO
    # -----------------------------
    def test_model(self, file_path):
        self.cluster_data(file_path, save_csv=True)
        df = pd.read_csv("database_clusters.csv")
        df = self._preprocess(df, fit=False)

        X = df.drop(columns=["Revenue"])
        y = df["Revenue"]

        y_pred = self.model.predict(X)
        print("==== Métricas de test ====")
        print("Tasa de error:", 1 - accuracy_score(y, y_pred))
        print("Precisión:", precision_score(y, y_pred))
        print("Recall:", recall_score(y, y_pred))
        print("F1-score:", f1_score(y, y_pred))

    # -----------------------------
    # CLUSTERING
    # -----------------------------
    def cluster_data(self, file_path, k=8, save_csv=True):
        df = pd.read_csv(file_path)

        num_cols = ['Administrative','Administrative_Duration','Informational',
                    'Informational_Duration','ProductRelated','ProductRelated_Duration',
                    'BounceRates','ExitRates','PageValues','SpecialDay']
        cat_cols = ['Month', 'VisitorType', 'Weekend',
                    'OperatingSystems','Browser','Region','TrafficType']

        for col in num_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])

        preprocessor = ColumnTransformer([
            ('num', Pipeline([('scaler', StandardScaler())]), [c for c in num_cols if c in df.columns]),
            ('cat', Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), 
             [c for c in cat_cols if c in df.columns])
        ])

        X = preprocessor.fit_transform(df)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        df['cluster'] = kmeans.fit_predict(X)

        X_dense = X.toarray() if hasattr(X, "toarray") else X
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_dense)
        df["PC1"] = X_pca[:, 0]
        df["PC2"] = X_pca[:, 1]

        if save_csv:
            df.to_csv("database_clusters.csv", index=False)
            print("\nArchivo generado: database_clusters.csv")

        self.preprocessor_cluster = preprocessor
        return df

    def cluster_group(self, df_group, k, label="Global"):
        X = self.preprocessor_cluster.transform(df_group)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        cluster_labels = kmeans.fit_predict(X)

        df_group['cluster'] = cluster_labels
        df_group['cluster_global'] = label
        df_group['cluster_id'] = df_group['cluster_global'] + "_" + df_group['cluster'].astype(str)

        X_dense = X.toarray() if hasattr(X, "toarray") else X
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_dense)
        df_group["PC1"] = X_pca[:, 0]
        df_group["PC2"] = X_pca[:, 1]

        # Visual opcional
        plt.figure(figsize=(8, 6))
        for c in range(k):
            pts = X_pca[cluster_labels == c]
            plt.scatter(pts[:, 0], pts[:, 1], alpha=0.6, label=f'Cluster {c}')
        plt.title(f'PCA – {label}')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.show()
    
        return df_group


# -----------------------------
# FLUJO DE DATOS
# -----------------------------
df = pd.read_csv("online_shoppers_forStudents.csv")

train_df, test_df = train_test_split(
    df, test_size=0.3, random_state=42, stratify=df["Revenue"]
)
train_df.to_csv("online_shoppers_train.csv", index=False)
test_df.to_csv("online_shoppers_test.csv", index=False)

# Entrenamiento y testeo
model = APS_Solver()
model.train_model("online_shoppers_train.csv")
model.test_model("online_shoppers_test.csv")
