from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.datasets import make_blobs

# Generate synthetic data (for demonstration)
X, _ = make_blobs(n_samples=1000, n_features=20, centers=2, random_state=42)

# 1. Isolation Forest
iso_forest = IsolationForest(contamination=0.1)
iso_forest.fit(X)
y_pred_iso = iso_forest.predict(X)  # -1 for outliers, 1 for inliers

# 2. One-Class SVM
one_class_svm = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
one_class_svm.fit(X)
y_pred_svm = one_class_svm.predict(X)  # -1 for outliers, 1 for inliers

# 3. Elliptic Envelope
elliptic_env = EllipticEnvelope(contamination=0.1)
elliptic_env.fit(X)
y_pred_elliptic = elliptic_env.predict(X)  # -1 for outliers, 1 for inliers

# 4. Local Outlier Factor (LOF)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred_lof = lof.fit_predict(X)  # -1 for outliers, 1 for inliers

# Display predictions for the first few samples (for demonstration)
print("Isolation Forest Predictions:", y_pred_iso[:10])
print("One-Class SVM Predictions:", y_pred_svm[:10])
print("Elliptic Envelope Predictions:", y_pred_elliptic[:10])
print("Local Outlier Factor Predictions:", y_pred_lof[:10])

print(X[:10])