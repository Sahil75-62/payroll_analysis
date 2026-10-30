
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score
)

warnings.filterwarnings("ignore")

# 1. Load Dataset
file_path = r"D:\MTA_Employee_Payroll__Beginning_2025.csv"
output_dir = r"D:\Payroll_ML_Project"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(file_path, low_memory=False)

print("Original Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())

# 2. Clean Dataset
df.columns = df.columns.str.strip()
df = df.drop_duplicates().copy()

numeric_cols = [
    "Hourly Rate", "Regular Pay", "Overtime Pay",
    "Cash Outs", "Retro Pay", "Other Pay", "Total Earnings"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

for col in ["Start Date", "Separation Date"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

if "Total Earnings" not in df.columns:
    raise ValueError("Total Earnings column is missing.")

if "Hourly Rate" not in df.columns:
    raise ValueError("Hourly Rate column is missing.")

if "Start Date" in df.columns:
    df["Start Year"] = df["Start Date"].dt.year
    df["Start Month"] = df["Start Date"].dt.month

    reference_date = pd.Timestamp("2025-12-31")
    df["Tenure Years Approx"] = (
        (reference_date - df["Start Date"]).dt.days / 365.25
    ).clip(lower=0)

if "Separation Date" in df.columns:
    df.loc[
        df["Separation Date"] == pd.Timestamp("1900-01-01"),
        "Separation Date"
    ] = pd.NaT

    df["Has Separation Date"] = (
        df["Separation Date"].notna().astype(int)
    )

df = df.dropna(subset=["Total Earnings"]).copy()
df = df[df["Total Earnings"] >= 0].copy()

cleaned_path = os.path.join(output_dir, "Payroll_Cleaned.csv")
df.to_csv(cleaned_path, index=False)

print("\nCleaned Shape:", df.shape)
print("\nMissing Values After Cleaning:\n", df.isnull().sum())
print("\nStatistical Summary:\n", df.describe(include="all").T)

# 3. Exploratory Data Analysis

plt.figure(figsize=(9, 5))
sns.histplot(df["Total Earnings"], bins=40, kde=True)
plt.title("Distribution of Total Earnings")
plt.xlabel("Total Earnings")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "earnings_distribution.png"))
plt.show()

plt.figure(figsize=(9, 4))
sns.boxplot(x=df["Total Earnings"])
plt.title("Total Earnings - Outlier Inspection")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "earnings_boxplot.png"))
plt.show()

if "Working Agency" in df.columns:
    agency_pay = (
        df.groupby("Working Agency")["Total Earnings"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    agency_pay.plot(kind="bar", figsize=(10, 5))
    plt.title("Average Total Earnings by Agency")
    plt.ylabel("Average Earnings")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "agency_earnings.png"))
    plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Hourly Rate",
    y="Total Earnings",
    alpha=0.35
)
plt.title("Hourly Rate vs Total Earnings")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "hourly_rate_scatter.png"))
plt.show()

corr_cols = [
    c for c in numeric_cols + ["Tenure Years Approx", "Start Year"]
    if c in df.columns
]

plt.figure(figsize=(10, 7))
sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Payroll Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
plt.show()

# 4. Prepare Data for Regression

num_features = [
    c for c in [
        "Hourly Rate", "Tenure Years Approx",
        "Start Year", "Start Month", "Has Separation Date"
    ] if c in df.columns
]

cat_features = [
    c for c in ["Working Agency", "Title", "Department", "Pay Basis"]
    if c in df.columns
]

X = df[num_features + cat_features].copy()
y = df["Total Earnings"].copy()

num_features = [c for c in num_features if X[c].notna().any()]
cat_features = [c for c in cat_features if X[c].notna().any()]

features = num_features + cat_features
X = X[features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def make_preprocessor(nums, cats):
    transformers = []

    if nums:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", numeric_pipe, nums))

    if cats:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=2
            ))
        ])
        transformers.append(("cat", categorical_pipe, cats))

    return ColumnTransformer(transformers)

results = []

def evaluate_model(name, model, Xtr, Xte, ytr, yte):
    model.fit(Xtr, ytr)
    predictions = model.predict(Xte)

    mae = mean_absolute_error(yte, predictions)
    mse = mean_squared_error(yte, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(yte, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print("\n", name)
    print("MAE:", round(mae, 2))
    print("MSE:", round(mse, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))

    return model, predictions

# 5. Model 1: Simple Linear Regression

simple_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

simple_model, simple_pred = evaluate_model(
    "Simple Linear Regression",
    simple_model,
    X_train[["Hourly Rate"]],
    X_test[["Hourly Rate"]],
    y_train,
    y_test
)

# Graph: Simple Linear Regression

plt.figure(figsize=(8, 5))
plt.scatter(y_test, simple_pred, alpha=0.5)
low = min(y_test.min(), simple_pred.min())
high = max(y_test.max(), simple_pred.max())
plt.plot([low, high], [low, high], "r--")
plt.xlabel("Actual Total Earnings")
plt.ylabel("Predicted Total Earnings")
plt.title("Simple Linear Regression")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "simple_linear_regression.png"))
plt.show()

# 6. Model 2: Multiple Linear Regression

multiple_model = Pipeline([
    ("preprocessor", make_preprocessor(num_features, cat_features)),
    ("regressor", LinearRegression())
])

multiple_model, multiple_pred = evaluate_model(
    "Multiple Linear Regression",
    multiple_model,
    X_train,
    X_test,
    y_train,
    y_test
)

# Graph: Multiple Linear Regression

plt.figure(figsize=(8, 5))
plt.scatter(y_test, multiple_pred, alpha=0.5)
low = min(y_test.min(), multiple_pred.min())
high = max(y_test.max(), multiple_pred.max())
plt.plot([low, high], [low, high], "r--")
plt.xlabel("Actual Total Earnings")
plt.ylabel("Predicted Total Earnings")
plt.title("Multiple Linear Regression")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "multiple_linear_regression.png"))
plt.show()

# 7. Model 3: Polynomial Regression

poly_cols = [
    c for c in ["Hourly Rate", "Tenure Years Approx"]
    if c in num_features
]

poly_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("polynomial", PolynomialFeatures(
        degree=2,
        include_bias=False
    )),
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

poly_model, poly_pred = evaluate_model(
    "Polynomial Regression",
    poly_model,
    X_train[poly_cols],
    X_test[poly_cols],
    y_train,
    y_test
)

# Graph: Polynomial Regression

plt.figure(figsize=(8, 5))
plt.scatter(y_test, poly_pred, alpha=0.5)
low = min(y_test.min(), poly_pred.min())
high = max(y_test.max(), poly_pred.max())
plt.plot([low, high], [low, high], "r--")
plt.xlabel("Actual Total Earnings")
plt.ylabel("Predicted Total Earnings")
plt.title("Polynomial Regression")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "polynomial_regression.png"))
plt.show()

# 8. Model 4: Random Forest Regression

forest_model = Pipeline([
    ("preprocessor", make_preprocessor(num_features, cat_features)),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

forest_model, forest_pred = evaluate_model(
    "Random Forest Regression",
    forest_model,
    X_train,
    X_test,
    y_train,
    y_test
)

# Graph: Random Forest Regression

plt.figure(figsize=(8, 5))
plt.scatter(y_test, forest_pred, alpha=0.5)
low = min(y_test.min(), forest_pred.min())
high = max(y_test.max(), forest_pred.max())
plt.plot([low, high], [low, high], "r--")
plt.xlabel("Actual Total Earnings")
plt.ylabel("Predicted Total Earnings")
plt.title("Random Forest Regression")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "random_forest_regression.png"))
plt.show()

# 9. Compare Regression Models

results_df = pd.DataFrame(results).sort_values("RMSE")

print("\nMODEL COMPARISON:\n")
print(results_df.to_string(index=False))

results_df.to_csv(
    os.path.join(output_dir, "model_comparison.csv"),
    index=False
)

results_df.set_index("Model")[["RMSE", "MAE"]].plot(
    kind="bar",
    figsize=(11, 5)
)
plt.title("Regression Model Performance")
plt.ylabel("Error (lower is better)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "model_comparison.png"))
plt.show()

# Graph: Best Regression Model

best_model_name = results_df.iloc[0]["Model"]

predictions_map = {
    "Simple Linear Regression": simple_pred,
    "Multiple Linear Regression": multiple_pred,
    "Polynomial Regression": poly_pred,
    "Random Forest Regression": forest_pred
}

best_predictions = predictions_map[best_model_name]

plt.figure(figsize=(7, 6))
plt.scatter(y_test, best_predictions, alpha=0.4)
low = min(y_test.min(), best_predictions.min())
high = max(y_test.max(), best_predictions.max())
plt.plot([low, high], [low, high], "r--")
plt.xlabel("Actual Total Earnings")
plt.ylabel("Predicted Total Earnings")
plt.title("Best Model: " + best_model_name)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "actual_vs_predicted.png"))
plt.show()

# 10. Model 5: K-Means Clustering

cluster_cols = [
    c for c in [
        "Hourly Rate", "Regular Pay",
        "Overtime Pay", "Tenure Years Approx"
    ] if c in df.columns
]

cluster_data = df[cluster_cols].copy()
cluster_data = cluster_data.replace([np.inf, -np.inf], np.nan)
cluster_data = cluster_data.dropna(how="all")

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

cluster_scaled = scaler.fit_transform(
    imputer.fit_transform(cluster_data)
)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(cluster_scaled)
df["Payroll Cluster"] = np.nan
df.loc[cluster_data.index, "Payroll Cluster"] = cluster_labels

if len(set(cluster_labels)) > 1 and len(cluster_labels) > 3:
    score = silhouette_score(cluster_scaled, cluster_labels)
    print("\nK-Means Silhouette Score:", round(score, 4))

print("\nCluster Summary:")
print(
    df.groupby("Payroll Cluster")[
        [c for c in ["Total Earnings", "Hourly Rate", "Overtime Pay"]
         if c in df.columns]
    ].mean()
)

# Graph: K-Means Clusters

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df.loc[cluster_data.index],
    x="Hourly Rate",
    y="Total Earnings",
    hue="Payroll Cluster",
    palette="Set2",
    alpha=0.6
)
plt.title("Employee Payroll Clusters")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "kmeans_clusters.png"))
plt.show()

# 11. Save Final Dataset

df.to_csv(
    os.path.join(output_dir, "Payroll_With_Clusters.csv"),
    index=False
)

print("\nProject completed successfully.")
print("Output folder:", output_dir)
