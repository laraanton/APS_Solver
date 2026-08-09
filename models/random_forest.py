import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import difflib

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class APS_Solver:

    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.preprocessor_cluster = None

    # ------------------------------------------------------
    # CODIFICACIÓN DE CATEGÓRICAS
    # ------------------------------------------------------
    def _encode_categorical(self, df, fit=True):
        df = df.copy()
        categorical_columns = df.select_dtypes(include=["object", "bool"]).columns

        for col in categorical_columns:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                df[col] = le.transform(df[col].astype(str))

        return df

    # ------------------------------------------------------
    # LIMPIEZA Y PREPROCESADO
    # ------------------------------------------------------
    def _preprocess(self, df, fit=True):
        df = df.copy().drop_duplicates()

        # -------- Eliminación de columnas irrelevantes --------
        drop_cols = ["OperatingSystems", "Browser", "Region", "TrafficType"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # Eliminación de valores NULL
        num_cols = [
            "Administrative","Administrative_Duration",
            "Informational","Informational_Duration",
            "ProductRelated","ProductRelated_Duration",
            "BounceRates","ExitRates","PageValues"
        ]
        #para las variables numericas: mediana
        for col in num_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        if "SpecialDay" in df.columns:
            df["SpecialDay"] = df["SpecialDay"].fillna(0)

        # -------- VisitorType --------
        if "VisitorType" in df.columns:
            df["VisitorType"] = (
                df["VisitorType"]
                .astype(str)
                .str.replace(" ", "_")
                .str.replace("-", "_")
                .str.strip()
            )
            visitor_map = {
                "Returning_Visitor": "Returning_Visitor",
                "New_Visitor": "New_Visitor",
                "Other": "Other",
                "returning_visitor": "Returning_Visitor",
                "new_visitor": "New_Visitor"
            }
            df["VisitorType"] = df["VisitorType"].replace(visitor_map).fillna("Other")

        # -------- Month --------
        if "Month" in df.columns:
            df["Month"] = df["Month"].astype(str).str.strip()
            df["Month"] = df["Month"].apply(self._correct_month)

        # -------- Booleanos --------
        for col in ["Weekend", "Revenue"]:
            if col in df.columns:
                df[col] = df[col].astype(bool)

        # -------- Rangos --------
        for col in ["BounceRates", "ExitRates"]:
            if col in df.columns:
                df[col] = df[col].clip(0,1)
        for col in ["Administrative_Duration","Informational_Duration","ProductRelated_Duration"]:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)

        # -------- One-Hot Encoding para Month y VisitorType --------
        cat_cols = []
        if "Month" in df.columns:
            cat_cols.append("Month")
        if "VisitorType" in df.columns:
            cat_cols.append("VisitorType")

        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        # -------- Escalado numérico --------
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "Revenue"]
        if fit:
            self.scaler = StandardScaler()
            df[num_cols] = self.scaler.fit_transform(df[num_cols])
        else:
            df[num_cols] = self.scaler.transform(df[num_cols])

        return df

    # ------------------------------------------------------
    # CORRECCIÓN DE MESES (difflib)
    # ------------------------------------------------------
    def _correct_month(self, val):
        valid_months_full = {
            "january": "Jan","february": "Feb","march": "Mar","april": "Apr",
            "may": "May","june": "Jun","july": "Jul","august": "Aug",
            "september": "Sep","october": "Oct","november": "Nov","december": "Dec"
        }
        val_clean = val.strip().lower()
        match = difflib.get_close_matches(val_clean, list(valid_months_full.keys()), n=1, cutoff=0.0)
        return valid_months_full[match[0]] if match else val_clean[:3].title()
    
    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            self.model, self.scaler, self.label_encoders = pickle.load(f)
    
    def save_model(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump((self.model, self.scaler, self.label_encoders), f)
    # ------------------------------------------------------
    # ENTRENAR MODELO
    # ------------------------------------------------------
    def train_model(self, file_path):
        df = pd.read_csv(file_path)
        df = self._preprocess(df, fit=True)

        X = df.drop(columns=["Revenue"])
        y = df["Revenue"].astype(int)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_val)
        print("==== Métricas de validación ====")
        print("Tasa de error:", 1 - accuracy_score(y_val, y_pred))
        print("Precisión:", precision_score(y_val, y_pred))
        print("Recall:", recall_score(y_val, y_pred))
        print("F1-score:", f1_score(y_val, y_pred))

    # ------------------------------------------------------
    # TEST FINAL
    # ------------------------------------------------------
    def test_model(self, file_path):
        df = pd.read_csv(file_path)
        df = self._preprocess(df, fit=False)

        X = df.drop(columns=["Revenue"])
        y = df["Revenue"].astype(int)

        y_pred = self.model.predict(X)
        print("==== Métricas de test ====")
        print("Tasa de error:", 1 - accuracy_score(y, y_pred))
        print("Precisión:", precision_score(y, y_pred))
        print("Recall:", recall_score(y, y_pred))
        print("F1-score:", f1_score(y, y_pred))

    # ------------------------------------------------------
    # CLUSTERING SEPARADO POR REVENUE
    # ------------------------------------------------------
    def cluster_data(self, file_path, k_true=4, k_false=4, save_excel=True):
        # Usar dataset preprocesado
        df = pd.read_csv(file_path)
        df_proc = self._preprocess(df, fit=True)
    
        # Separar por Revenue
        df_true = df_proc[df_proc["Revenue"]==True].copy()
        df_false = df_proc[df_proc["Revenue"]==False].copy()
    
        def cluster_group(df_group, k, label):
            # Usar todas las columnas excepto Revenue
            X = df_group.drop(columns=["Revenue"]).values
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
            sub_labels = kmeans.fit_predict(X)
            df_group['cluster_sub'] = sub_labels
            df_group['cluster_global'] = label
    
            # PCA para visualización
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X)
            df_group["PC1"] = X_pca[:,0]
            df_group["PC2"] = X_pca[:,1]
    
            plt.figure(figsize=(8,6))
            for c in range(k):
                pts = X_pca[sub_labels==c]
                plt.scatter(pts[:,0], pts[:,1], alpha=0.6, label=f'Subcluster {c}')
            plt.title(f'PCA – {label}')
            plt.xlabel('PC1'); plt.ylabel('PC2'); plt.legend()
            plt.show()
    
            return df_group
    
        df_true = cluster_group(df_true, k_true, "Revenue_True")
        df_false = cluster_group(df_false, k_false, "Revenue_False")
    
        df_out = pd.concat([df_true, df_false]).sort_index()
    
        if save_excel:
            df_out.to_excel("database_clusters.xlsx", index=False)
            print("\nArchivo generado: database_clusters.xlsx")
    
        return df_out
    
# ============================================================
#  USO
# ============================================================

df = pd.read_csv("online_shoppers_forStudents.csv")
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["Revenue"])
train_df.to_csv("online_shoppers_train.csv", index=False)
test_df.to_csv("online_shoppers_test.csv", index=False)

model = APS_Solver()
print('Para el train:')
model.train_model("online_shoppers_train.csv")
print('\nPara el test:')
model.test_model("online_shoppers_test.csv")

# Llamada al clustering (usa archivo Excel original)
model.cluster_data("online_shoppers_forStudents.csv", k_true=4, k_false=4)
