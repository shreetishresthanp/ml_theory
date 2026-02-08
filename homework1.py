import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# Task 1: Generate Synthetic Data for Two Transfer Learning Tasks
# ============================================================================

def generate_transfer_learning_tasks(n_samples=200, p=100, noise_std=0.1, seed=42, delta=0.0):
    """
    Generate two synthetic tasks for transfer learning.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples for each task
    p : int
        Dimension of feature vectors (default: 100)
    noise_std : float
        Standard deviation of Gaussian noise (default: 0.1)
    seed : int
        Random seed for reproducibility
    delta : float
        Distance between beta(1) and beta(2). beta(2) = beta(1) + delta * direction
    
    Returns:
    --------
    X1, y1 : Features and labels for Task 1
    X2, y2 : Features and labels for Task 2
    beta1, beta2 : True coefficient vectors for Task 1 and Task 2
    """
    
    np.random.seed(seed)
    
    # Generate feature vectors from isotropic Gaussian N(0, I_p)
    X1 = np.random.randn(n_samples, p)
    X2 = np.random.randn(n_samples, p)
    
    # Generate true coefficient vector beta(1)
    beta1 = np.random.randn(p) * 0.5  # Scale down for reasonable signal-to-noise ratio
    
    # Generate beta(2) = beta(1) + delta * direction (random unit vector)
    direction = np.random.randn(p)
    direction = direction / np.linalg.norm(direction)
    beta2 = beta1 + delta * direction
    
    # Generate labels with Gaussian noise
    # y = X @ beta + noise, where noise ~ N(0, noise_std^2)
    noise1 = np.random.normal(0, noise_std, n_samples)
    noise2 = np.random.normal(0, noise_std, n_samples)
    
    y1 = X1 @ beta1 + noise1
    y2 = X2 @ beta2 + noise2
    
    return X1, y1, X2, y2, beta1, beta2


# ============================================================================
# Task 2: OLS vs HPS Estimators with Varying Task Distance δ
# ============================================================================

def train_test_split(X, y, test_ratio=0.3, seed=None):
    """Split data into train and test sets."""
    if seed is not None:
        np.random.seed(seed)
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)
    split_idx = int(n * (1 - test_ratio))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def ols_estimator(X_train, y_train):
    """
    Ordinary Least Squares (OLS) estimator.
    Fits: beta = argmin || y - X @ beta ||^2
    """
    model = LinearRegression(fit_intercept=False)
    model.fit(X_train, y_train)
    return model.coef_


def hps_estimator(X1_train, y1_train, X2_train, y2_train):
    """
    Hard Parameter Sharing (HPS) estimator.
    Fits shared beta by minimizing: || y1 - X1 @ beta ||^2 + || y2 - X2 @ beta ||^2
    """
    # Stack the data from both tasks
    X_combined = np.vstack([X1_train, X2_train])
    y_combined = np.hstack([y1_train, y2_train])
    
    # Fit shared parameter
    model = LinearRegression(fit_intercept=False)
    model.fit(X_combined, y_combined)
    return model.coef_


def compute_test_loss(X_test, y_test, beta):
    """Compute MSE on test set."""
    y_pred = X_test @ beta
    return np.mean((y_test - y_pred) ** 2)


# Experiment parameters
n1 = 200  # Task 1: 200 samples
n2 = 100  # Task 2: 100 samples
p = 100   # Feature dimension
noise_std = 0.1
test_ratio = 0.3

# Vary delta from 0.01 to 1.00
delta_values = np.linspace(0.01, 1.00, 20)
ols_losses = []
hps_losses = []

print("=" * 70)
print("TRANSFER LEARNING: OLS vs HPS with varying δ")
print("=" * 70)
print(f"\nSetup:")
print(f"  Task 1 samples: {n1}")
print(f"  Task 2 samples: {n2}")
print(f"  Feature dimension: {p}")
print(f"  Noise std: {noise_std}")
print(f"  Test ratio: {test_ratio:.1%}")
print(f"\nVarying delta from {delta_values[0]:.2f} to {delta_values[-1]:.2f}...")

# Run experiment over delta values
for delta in delta_values:
    # Generate synthetic data with this delta value
    X1, y1, X2, y2, beta1_true, beta2_true = generate_transfer_learning_tasks(
        n_samples=n1,
        p=p,
        noise_std=noise_std,
        seed=42,
        delta=delta
    )
    
    # For Task 2, we'll evaluate both approaches
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_ratio=test_ratio, seed=42)
    
    # OLS: Train only on Task 2
    beta_ols = ols_estimator(X2_train, y2_train)
    loss_ols = compute_test_loss(X2_test, y2_test, beta_ols)
    ols_losses.append(loss_ols)
    
    # HPS: Train on Task 1 (full data) and Task 2 training set
    beta_hps = hps_estimator(X1, y1, X2_train, y2_train)
    loss_hps = compute_test_loss(X2_test, y2_test, beta_hps)
    hps_losses.append(loss_hps)

print(f"\nExperiment complete!")
print(f"\n{'Delta':>8} {'OLS Loss':>12} {'HPS Loss':>12} {'HPS Improvement':>15}")
print("-" * 55)
for delta, loss_ols, loss_hps in zip(delta_values, ols_losses, hps_losses):
    improvement = (loss_ols - loss_hps) / loss_ols * 100
    print(f"{delta:8.4f} {loss_ols:12.6f} {loss_hps:12.6f} {improvement:14.2f}%")

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(delta_values, ols_losses, 'o-', linewidth=2, markersize=6, label='OLS (Single Task)', color='#1f77b4')
plt.plot(delta_values, hps_losses, 's-', linewidth=2, markersize=6, label='HPS (Transfer Learning)', color='#ff7f0e')
plt.xlabel('Task Distance (δ)', fontsize=12)
plt.ylabel('Test Loss (MSE)', fontsize=12)
plt.title(f'Transfer Learning Comparison: OLS vs HPS\n(n₁={n1}, n₂={n2}, p={p})', fontsize=13)
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('transfer_learning_comparison.png', dpi=150, bbox_inches='tight')
print("\nPlot saved as 'transfer_learning_comparison.png'")
plt.show()

print("\n" + "=" * 70)
