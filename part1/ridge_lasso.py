from ols_implementation import mat_transpose, mat_mul, mat_inverse, mat_zeros, mat_vec_mul

def ridge_fit(X, y, lam):
    """
    Fit Ridge Regression with closed-form solution.

    The intercept is NOT penalised (I_mod[0, 0] = 0).
    """
    X = [list(row) for row in X]
    y = list(y)
    n = len(X)
    p = len(X[0])
    
    I_mod = mat_zeros(p, p)
    for i in range(1, p):
        I_mod[i][i] = 1.0
        
    XT = mat_transpose(X)
    XTX = mat_mul(XT, X)
    
    for i in range(p):
        for j in range(p):
            XTX[i][j] += lam * I_mod[i][j]
            
    XTX_inv = mat_inverse(XTX)
    y_col = [[yi] for yi in y]
    XTy = mat_mul(XT, y_col)
    XTy_vec = [r[0] for r in XTy]
    
    beta_hat = mat_vec_mul(XTX_inv, XTy_vec)
    return beta_hat
