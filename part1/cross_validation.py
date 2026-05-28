import random
from ols_implementation import mat_transpose, mat_mul, mat_inverse, mat_vec_mul

def kfold_cv(X, y, k=5, random_state=42):
    """
    K-fold cross-validation for OLS regression.
    """
    X = [list(row) for row in X]
    y = list(y)
    n = len(y)
    
    rng = random.Random(random_state)
    indices = list(range(n))
    rng.shuffle(indices)
    
    fold_sizes = [n // k] * k
    for i in range(n % k):
        fold_sizes[i] += 1
        
    current = 0
    folds = []
    for fold_size in fold_sizes:
        stop = current + fold_size
        folds.append(indices[current:stop])
        current = stop
        
    mse_scores = []
    for i in range(k):
        test_indices = folds[i]
        train_indices = []
        for j in range(k):
            if j != i:
                train_indices.extend(folds[j])
                
        X_train = [X[idx] for idx in train_indices]
        y_train = [y[idx] for idx in train_indices]
        X_test = [X[idx] for idx in test_indices]
        y_test = [y[idx] for idx in test_indices]
        
        XT = mat_transpose(X_train)
        XTX = mat_mul(XT, X_train)
        XTX_inv = mat_inverse(XTX)
        y_col = [[yi] for yi in y_train]
        XTy = mat_mul(XT, y_col)
        XTy_vec = [r[0] for r in XTy]
        beta = mat_vec_mul(XTX_inv, XTy_vec)
        
        y_pred = mat_vec_mul(X_test, beta)
        mse = sum((y_test[j] - y_pred[j]) ** 2 for j in range(len(y_test))) / len(y_test)
        mse_scores.append(mse)
        
    mean_mse = sum(mse_scores) / k
    return mean_mse, mse_scores
