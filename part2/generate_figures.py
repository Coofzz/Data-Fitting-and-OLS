"""Generate and save all report figures to report/figures/."""
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import (
    load_raw, filter_and_winsorize, make_target, DataPipeline,
    FEATURES, EDLEVEL_MAP, WINSOR_UPPER
)
from sklearn.model_selection import train_test_split

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'report', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({'figure.dpi': 120, 'font.size': 10,
                     'axes.titlesize': 11, 'axes.labelsize': 10})

print("Loading data...")
df_raw = load_raw()
df_filt = filter_and_winsorize(df_raw)
y_all = make_target(df_filt)
df_feat = df_filt.drop(columns=['ConvertedCompYearly'])

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    df_feat, y_all, test_size=0.20, random_state=42
)
pipeline = DataPipeline()
X_train = pipeline.fit_transform(X_train_raw)
X_test  = pipeline.transform(X_test_raw)
mask_tr = ~X_train.isnull().any(axis=1)
mask_te = ~X_test.isnull().any(axis=1)
X_train, y_train = X_train[mask_tr], y_train_raw[mask_tr]
X_test,  y_test  = X_test[mask_te],  y_test_raw[mask_te]
features = list(X_train.columns)

# ── Fig 1: Missing rate bar chart ────────────────────────────────────────────
print("Fig 1: Missing rate...")
missing = df_raw.isnull().mean() * 100
fig, ax = plt.subplots(figsize=(9, 4))
colors = ['#e74c3c' if v >= 5 else '#3498db' for v in missing]
missing.sort_values(ascending=False).plot(kind='bar', ax=ax, color=colors, edgecolor='black')
ax.axhline(5, color='red', linestyle='--', linewidth=1.2, label='5% threshold')
ax.set_title('Missing Rate by Column (%)\nRed bars = above 5% threshold')
ax.set_ylabel('Missing (%)')
ax.set_xlabel('')
ax.legend()
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig1_missing_rate.png'))
plt.close()

# ── Fig 2: Salary distribution — before vs after log1p ───────────────────────
print("Fig 2: Salary distribution...")
sal_raw  = df_raw['ConvertedCompYearly'].dropna()
sal_wins = sal_raw.clip(0, WINSOR_UPPER)
sal_log  = np.log1p(sal_wins)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].hist(sal_raw.clip(0, 500_000), bins=60, color='#e74c3c', edgecolor='black', alpha=0.8)
axes[0].set_title('Raw salary\n(clipped at $500K for display)')
axes[0].set_xlabel('USD/year')
axes[0].set_ylabel('Count')

axes[1].hist(sal_wins, bins=60, color='#e67e22', edgecolor='black', alpha=0.8)
axes[1].axvline(WINSOR_UPPER, color='red', linestyle='--', label=f'Cap = ${WINSOR_UPPER:,}')
axes[1].set_title(f'After Winsorization\n(cap = ${WINSOR_UPPER:,})')
axes[1].set_xlabel('USD/year')
axes[1].legend()

axes[2].hist(sal_log, bins=60, color='#27ae60', edgecolor='black', alpha=0.8)
axes[2].set_title('After log1p transform\n(target variable y)')
axes[2].set_xlabel('log(1 + salary)')

plt.suptitle('Salary Distribution Transformation Pipeline', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig2_salary_transform.png'))
plt.close()

# ── Fig 3: IQR vs Z-score outlier detection ──────────────────────────────────
print("Fig 3: Outlier detection...")
Q1, Q3 = sal_raw.quantile(0.25), sal_raw.quantile(0.75)
IQR = Q3 - Q1
iqr_upper = Q3 + 1.5 * IQR
z_upper = sal_raw.mean() + 3 * sal_raw.std()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# Boxplot
axes[0].boxplot(sal_raw.clip(0, 600_000), vert=False,
                flierprops=dict(marker='.', alpha=0.2, markersize=3))
axes[0].axvline(iqr_upper, color='red', linestyle='--', linewidth=1.5,
                label=f'IQR upper = ${iqr_upper:,.0f}\nOutliers: {(sal_raw > iqr_upper).mean()*100:.1f}%')
axes[0].axvline(z_upper, color='blue', linestyle=':', linewidth=1.5,
                label=f'3σ upper = ${z_upper:,.0f}\nOutliers: {(sal_raw > z_upper).mean()*100:.1f}%')
axes[0].set_title('Outlier Thresholds: IQR vs Z-score')
axes[0].set_xlabel('USD/year (clipped at $600K)')
axes[0].legend(fontsize=8)

# Comparison bar
methods = ['IQR (1.5×IQR)', 'Z-score (3σ)']
pcts = [(sal_raw > iqr_upper).mean()*100, (sal_raw > z_upper).mean()*100]
bars = axes[1].bar(methods, pcts, color=['#e74c3c', '#3498db'], edgecolor='black', width=0.4)
for bar, pct in zip(bars, pcts):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{pct:.1f}%', ha='center', fontsize=11, fontweight='bold')
axes[1].set_title('% Flagged as Outlier\nby Each Method')
axes[1].set_ylabel('% of total salary responses')
axes[1].set_ylim(0, 7)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig3_outlier_detection.png'))
plt.close()

# ── Fig 4: Correlation heatmap ───────────────────────────────────────────────
print("Fig 4: Correlation heatmap...")
num_df = df_raw[['ConvertedCompYearly', 'YearsCodePro', 'WorkExp']].copy()
num_df['YearsCodePro'] = pd.to_numeric(
    num_df['YearsCodePro'].replace({'Less than 1 year': 0, 'More than 50 years': 51}),
    errors='coerce'
)
corr = num_df.dropna().corr()
fig, ax = plt.subplots(figsize=(5, 4))
mask = np.zeros_like(corr, dtype=bool)
sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            ax=ax, linewidths=0.5, annot_kws={'size': 12})
ax.set_title('Pearson Correlation Matrix\n(numeric features only)')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig4_correlation_heatmap.png'))
plt.close()

# ── Fig 5: DataPipeline fit/transform diagram ────────────────────────────────
print("Fig 5: Pipeline diagram...")
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.axis('off')
steps = [
    ("Raw Data\n89,184 rows\n10 columns", '#e74c3c'),
    ("Listwise deletion\n(MNAR: salary)\n→ 48,019 rows", '#e67e22'),
    ("Winsorize\ncap = $238K\n+ log1p(y)", '#f1c40f'),
    ("Encode features\nordinal / binary\none-hot", '#2ecc71'),
    ("Train/Test split\n80% / 20%\nrandom_state=42", '#3498db'),
    ("StandardScaler\nfit on TRAIN only\ntransform TEST", '#9b59b6'),
    ("Clean Dataset\n38,415 train\n9,604 test | 38 feat", '#1abc9c'),
]
box_w, box_h = 0.12, 0.55
gap = 0.135
for i, (text, color) in enumerate(steps):
    x = 0.02 + i * gap
    ax.add_patch(plt.Rectangle((x, 0.2), box_w, box_h, color=color, alpha=0.85,
                                transform=ax.transAxes, clip_on=False))
    ax.text(x + box_w/2, 0.475, text, transform=ax.transAxes,
            ha='center', va='center', fontsize=7.5, fontweight='bold', color='white',
            multialignment='center')
    if i < len(steps) - 1:
        ax.annotate('', xy=(x + box_w + 0.003, 0.475), xytext=(x + box_w, 0.475),
                    xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))

ax.set_title('DataPipeline — Full Preprocessing Flow', fontsize=12, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig5_pipeline_diagram.png'))
plt.close()

# ── Fig 6: Model comparison bar chart ────────────────────────────────────────
print("Fig 6: Model comparison...")
import statsmodels.api as sm

def get_metrics(name, y_true, y_pred):
    return {'name': name,
            'MAE':  mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2':   r2_score(y_true, y_pred)}

# M1 OLS Full
Xc = sm.add_constant(X_train); m1 = sm.OLS(y_train, Xc).fit()
r1 = get_metrics('OLS Full', y_test, m1.predict(sm.add_constant(X_test)))

# M3 Ridge
ridge = RidgeCV(alphas=np.logspace(-3,3,100), cv=5).fit(X_train, y_train)
r3 = get_metrics('Ridge', y_test, ridge.predict(X_test))

# M3 Lasso
lasso = LassoCV(alphas=np.logspace(-3,3,100), cv=5, max_iter=10000, random_state=42).fit(X_train, y_train)
r4 = get_metrics('Lasso', y_test, lasso.predict(X_test))

# M4 Poly+Ridge
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2, include_bias=False)
Xp_tr = poly.fit_transform(X_train); Xp_te = poly.transform(X_test)
pr = RidgeCV(alphas=np.logspace(-3,3,50), cv=5).fit(Xp_tr, y_train)
r5 = get_metrics('Poly+Ridge', y_test, pr.predict(Xp_te))

results = [r1, r3, r4, r5]
names = [r['name'] for r in results]
r2s   = [r['R2']   for r in results]
maes  = [r['MAE']  for r in results]
rmses = [r['RMSE'] for r in results]

x = np.arange(len(names))
w = 0.25
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

bars = axes[0].bar(x, r2s, width=0.5, color=['#3498db','#e74c3c','#e67e22','#27ae60'],
                   edgecolor='black')
for bar, v in zip(bars, r2s):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                 f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=15)
axes[0].set_ylabel('R² (higher is better)')
axes[0].set_title('R² on Test Set')
axes[0].set_ylim(0.35, 0.46)

axes[1].bar(x - w/2, maes,  width=w, label='MAE',  color='#3498db', edgecolor='black')
axes[1].bar(x + w/2, rmses, width=w, label='RMSE', color='#e74c3c', edgecolor='black', alpha=0.85)
axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=15)
axes[1].set_ylabel('Error (log salary scale)')
axes[1].set_title('MAE & RMSE on Test Set')
axes[1].legend()

plt.suptitle('Model Comparison — Test Set Performance', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig6_model_comparison.png'))
plt.close()

# ── Fig 7: Residual analysis (Ridge) ─────────────────────────────────────────
print("Fig 7: Residual plots...")
y_pred_r = ridge.predict(X_test)
residuals = np.asarray(y_test) - y_pred_r

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Residual Diagnostic Plots — Ridge Regression (Best Model)',
             fontsize=13, fontweight='bold')

axes[0,0].scatter(y_pred_r, residuals, alpha=0.15, s=5, color='steelblue')
axes[0,0].axhline(0, color='red', lw=1.5, linestyle='--')
axes[0,0].set_xlabel('Fitted values (log salary)')
axes[0,0].set_ylabel('Residuals')
axes[0,0].set_title('Residuals vs Fitted')

stats.probplot(residuals, dist='norm', plot=axes[0,1])
axes[0,1].set_title('Normal Q-Q Plot')

axes[1,0].scatter(y_pred_r, np.sqrt(np.abs(residuals)), alpha=0.15, s=5, color='darkorange')
axes[1,0].set_xlabel('Fitted values (log salary)')
axes[1,0].set_ylabel('√|Residuals|')
axes[1,0].set_title('Scale-Location (Homoscedasticity Check)')

axes[1,1].hist(residuals, bins=60, edgecolor='black', alpha=0.75, color='steelblue')
xr = np.linspace(residuals.min(), residuals.max(), 300)
axes[1,1].plot(xr, stats.norm.pdf(xr, residuals.mean(), residuals.std()) *
               len(residuals) * (residuals.max()-residuals.min()) / 60,
               color='red', lw=2, label='Normal fit')
axes[1,1].set_xlabel('Residual'); axes[1,1].set_ylabel('Count')
axes[1,1].set_title('Residual Distribution')
axes[1,1].legend()

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig7_residual_plots.png'))
plt.close()

# ── Fig 8: Feature importance (Ridge) ────────────────────────────────────────
print("Fig 8: Feature importance...")
coef_s = pd.Series(np.abs(ridge.coef_), index=features).sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(9, 6))
colors_fi = ['#e74c3c' if 'Country' in f else
             '#3498db' if f in ['YearsCodePro','WorkExp','EdLevel','OrgSize'] else
             '#27ae60' for f in coef_s.sort_values().index]
coef_s.sort_values().plot(kind='barh', ax=ax, color=colors_fi, edgecolor='black')
ax.set_xlabel('|Coefficient| (standardized scale)')
ax.set_title('Feature Importance — Ridge Regression (Top 20)\n'
             'Red=Country, Blue=Numeric, Green=Other')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#e74c3c', label='Country features'),
                   Patch(color='#3498db', label='Numeric features'),
                   Patch(color='#27ae60', label='Other features')], loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig8_feature_importance.png'))
plt.close()

# ── Fig 9: Ridge trace (alpha selection) ─────────────────────────────────────
print("Fig 9: Ridge CV alpha selection...")
alphas_range = np.logspace(-3, 3, 200)
cv_scores = []
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
for a in alphas_range[::5]:
    sc = cross_val_score(Ridge(alpha=a), X_train, y_train, cv=5,
                         scoring='neg_mean_squared_error')
    cv_scores.append(-sc.mean())
alpha_plot = alphas_range[::5]

fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogx(alpha_plot, cv_scores, 'b-o', markersize=3)
best_idx = np.argmin(cv_scores)
ax.axvline(ridge.alpha_, color='red', linestyle='--',
           label=f'Best α = {ridge.alpha_:.4f}')
ax.scatter([alpha_plot[best_idx]], [cv_scores[best_idx]], color='red', s=60, zorder=5)
ax.set_xlabel('Alpha (regularization strength) — log scale')
ax.set_ylabel('5-fold CV MSE')
ax.set_title('Ridge CV — Alpha Selection')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig9_ridge_alpha_cv.png'))
plt.close()

print(f"\nAll figures saved to: {os.path.abspath(FIG_DIR)}")
print("Files:", sorted(os.listdir(FIG_DIR)))
