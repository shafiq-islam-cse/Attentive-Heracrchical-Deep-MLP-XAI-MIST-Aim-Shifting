import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats

# ---------------- LOAD DATA ----------------
drive.mount('/content/drive')

file_path = '/content/drive/MyDrive/Research Documents/Updated Work/MIST Level 3 Research Group Career Shifting After COVID-19/New analysis 2025/MIST-Career-Shift-Data-2025.csv'

df = pd.read_csv(file_path)

print('Raw data')
print(df)

# Replace inf with NaN, then fill NaN with column 0
df = df.replace([np.inf, -np.inf], np.nan)

# Target column handling
target_col = df.columns[-1]
df[target_col] = df[target_col].fillna('Aim has been shifted')

# Get unique values from that column
unique_classes = df[target_col].unique()
print("All target class names are:")
print(unique_classes)

df = df.fillna(0)

# Display dataset info
print("First 5 rows of the dataset:")
display(df.head())
print("\nShape after loading:", df.shape)

# Drop unnecessary columns
columns_to_drop = ['Timestamp', 'Username', 'Name (নাম)', 'Name (নাম)', 'Email (ই-মেইল)', 'Current Institution Name','Email']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# ---------------- RENAME FEATURES TO F1, F2... ----------------
# Identify feature columns (all columns except target)
feature_cols = [col for col in df.columns if col != target_col]

# Create generic names F1, F2, ..., Fn
generic_feature_names = [f'F{i+1}' for i in range(len(feature_cols))]

# Create mapping dictionary (only mapping features, leaving target alone)
name_mapping = {old: new for old, new in zip(feature_cols, generic_feature_names)}

# Apply renaming to the dataframe
df = df.rename(columns=name_mapping)

print("Columns after renaming to generic F-names:")
print(df.columns.tolist())
print("\nShape after renaming:", df.shape)


# ---------------- Basic Feature Statistics ----------------
summary_list = []

for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:  # numeric columns
        mean_val = np.nanmean(df[col])
        std_val = np.nanstd(df[col], ddof=1)
        median_val = np.nanmedian(df[col])
        min_val = np.nanmin(df[col])
        max_val = np.nanmax(df[col])

        if df[col].notna().sum() > 1:
            t_stat, p_val = stats.ttest_1samp(df[col], 0, nan_policy='omit')
        else:
            t_stat, p_val = np.nan, np.nan

        summary_list.append([col, 'numeric', mean_val, std_val, median_val, min_val, max_val, t_stat, p_val])

    else:  # categorical columns
        counts = df[col].value_counts(dropna=False)
        percentages = df[col].value_counts(normalize=True, dropna=False) * 100
        summary_list.append([col, 'categorical', counts.to_dict(), percentages.to_dict(),
                             np.nan, np.nan, np.nan, np.nan, np.nan])

summary_df = pd.DataFrame(summary_list, columns=[
    'Feature', 'Type', 'Counts/Mean', 'Percentages/Std',
    'Median', 'Min', 'Max', 'T-Value', 'P-Value'
])

pd.set_option('display.max_columns', None)
display(summary_df)

# Convert categorical strings to integer codes
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype('category').cat.codes

print("\nData types after categorical encoding:")
display(df.dtypes)

print("\nMissing values per column:")
display(df.isnull().sum())

# ---------------- FEATURE CORRELATION BEFORE SMOTE ----------------
print("\n=== Feature Correlation Analysis Before SMOTE ===")

# Create a temporary copy to rename the target to 'T' for the plot
corr_df_before = df.copy()
corr_df_before = corr_df_before.rename(columns={target_col: 'T'})

corr_before = corr_df_before.corr()

plt.figure(figsize=(32, 24))
mask = np.triu(np.ones_like(corr_before, dtype=bool))
sns.heatmap(
    corr_before,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=.5,
    cbar_kws={"shrink": .5},
    annot_kws={"size": 25} # Set to 25
)
plt.title('Feature Correlation Matrix Before SMOTE', fontsize=25, pad=20)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.tight_layout()
plt.savefig("Figure Feature Correlation Before SMOTE.pdf", dpi=100)
plt.show()

# ---------------- OLS REGRESSION BEFORE SMOTE ----------------
print("\n=== OLS Regression Model Before SMOTE ===")
X_before = df.drop(target_col, axis=1)
y_before = df[target_col]

print("\nSummary statistics raw(Before SMOTE):")
display(X_before.describe())

X_before_const = sm.add_constant(X_before)
ols_before = sm.OLS(y_before, X_before_const).fit()

print("OLS Regression Summary Before SMOTE:")
print(ols_before.summary())

plt.figure(figsize=(14, 10)) # Increased size slightly for larger fonts
coef_before = ols_before.params.drop('const').sort_values(ascending=False)
ax = sns.barplot(x=coef_before.values, y=coef_before.index, palette="viridis")
plt.title('OLS Regression Coefficients Before SMOTE', fontsize=25, pad=20)
plt.xlabel('Coefficient Value', fontsize=25)
plt.ylabel('Features', fontsize=25)
plt.yticks(fontsize=25) # Sets "F1, F2..." text to 25

# Add value labels on the bars
for i, v in enumerate(coef_before.values):
    ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=25)

plt.tight_layout()
plt.savefig("Figure OLS Coefficients Before SMOTE.pdf", dpi=100)
plt.show()

# ---------------- TARGET SPLIT ----------------
X = df.drop(target_col, axis=1)
y = df[target_col]

print("\nOriginal class distribution:")
print(y.value_counts())

plt.figure(figsize=(8, 6))
colors = plt.cm.Set3(np.linspace(0, 1, len(y.value_counts())))
explode = [0.05] * len(y.value_counts())

def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return f'{pct:.1f}%\n({val})'
    return my_format

plt.pie(y.value_counts(),
        labels=y.value_counts().index,
        autopct=autopct_format(y.value_counts()),
        colors=colors,
        explode=explode,
        shadow=True,
        startangle=90,
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
        textprops={'fontsize': 12})

centre_circle = plt.Circle((0,0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title("Original Class Distribution", fontsize=16, pad=20)
plt.axis('equal')
plt.tight_layout()
plt.draw()
plt.savefig("Figure Original Class Distribution.pdf", dpi=100)
plt.show()

# t-SNE visualization before SMOTE
print("\nGenerating t-SNE visualization before SMOTE...")
plt.figure(figsize=(8, 6))

tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X)

unique_classes = y.unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))

for i, class_label in enumerate(unique_classes):
    plt.scatter(X_tsne[y == class_label, 0],
                X_tsne[y == class_label, 1],
                c=[colors[i]],
                label=f'Class {class_label}',
                alpha=0.7,
                s=30)

plt.title('t-SNE Visualization Before SMOTE', fontsize=16, pad=20)
plt.xlabel('t-SNE Component 1', fontsize=12)
plt.ylabel('t-SNE Component 2', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Figure t-SNE Before SMOTE.pdf", dpi=100)
plt.show()

# ---------------- APPLY SMOTE ----------------
desired_count = 500
class_counts = y.value_counts()
sampling_strategy = {cls: desired_count for cls in class_counts.index if class_counts[cls] < desired_count}

smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nNew class distribution after SMOTE:")
print(y_resampled.value_counts())

plt.figure(figsize=(8, 6))
colors = plt.cm.Set3(np.linspace(0, 1, len(y_resampled.value_counts())))
explode = [0.05] * len(y_resampled.value_counts())

plt.pie(y_resampled.value_counts(),
        labels=y_resampled.value_counts().index,
        autopct=autopct_format(y_resampled.value_counts()),
        colors=colors,
        explode=explode,
        shadow=True,
        startangle=90,
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
        textprops={'fontsize': 12})

centre_circle = plt.Circle((0,0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title("Class Distribution After SMOTE", fontsize=16, pad=20)
plt.axis('equal')
plt.tight_layout()
plt.draw()
plt.savefig("Figure Class Distribution After SMOTE.pdf", dpi=100)
plt.show()

df_resampled = pd.concat([X_resampled, y_resampled], axis=1)

# ---------------- FEATURE CORRELATION AFTER SMOTE ----------------
print("\n=== Feature Correlation Analysis After SMOTE ===")

# Create a temporary copy to rename the target to 'T' for the plot
corr_df_after = df_resampled.copy()
corr_df_after = corr_df_after.rename(columns={target_col: 'T'})

corr_after = corr_df_after.corr()

plt.figure(figsize=(32, 24))
mask = np.triu(np.ones_like(corr_after, dtype=bool))

sns.heatmap(
    corr_after,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=.5,
    cbar_kws={"shrink": .5},
    annot_kws={"size": 25} # Set to 25
)
plt.title('Feature Correlation Matrix After SMOTE', fontsize=25, pad=20)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.tight_layout()
plt.savefig("Figure Feature Correlation After SMOTE.pdf", dpi=100)
plt.show()

# ---------------- OLS REGRESSION AFTER SMOTE ----------------
print("\n=== OLS Regression Model After SMOTE ===")
X_after = df_resampled.drop(target_col, axis=1)
y_after = df_resampled[target_col]

X_after_const = sm.add_constant(X_after)
ols_after = sm.OLS(y_after, X_after_const).fit()

print("OLS Regression Summary After SMOTE:")
print(ols_after.summary())

plt.figure(figsize=(14, 10)) # Increased size slightly for larger fonts
coef_after = ols_after.params.drop('const').sort_values(ascending=False)
ax = sns.barplot(x=coef_after.values, y=coef_after.index, palette="viridis")
plt.title('OLS Regression Coefficients After SMOTE', fontsize=25, pad=20)
plt.xlabel('Coefficient Value', fontsize=25)
plt.ylabel('Features', fontsize=25)
plt.yticks(fontsize=25) # Sets "F1, F2..." text to 25

# Add value labels on the bars
for i, v in enumerate(coef_after.values):
    ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=25)

plt.tight_layout()
plt.savefig("Figure OLS Coefficients After SMOTE.pdf", dpi=100)
plt.show()

print("\nSummary statistics (After SMOTE):")
display(X_resampled.describe())

plt.figure(figsize=(12, 8))
desc_stats = X_resampled.describe()
sns.heatmap(desc_stats, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True,
            linewidths=0.5, linecolor='white')
plt.title('Descriptive Statistics After SMOTE', fontsize=16, pad=20)
plt.xlabel('Statistics', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
plt.savefig("Figure Descriptive Statistics After SMOTE.pdf", dpi=100)
plt.show()

plt.figure(figsize=(10, 6))
mean_values = X_resampled.mean().sort_values(ascending=False)
ax = sns.barplot(x=mean_values.values, y=mean_values.index, palette="viridis")
plt.title('Mean Values of Features After SMOTE', fontsize=16, pad=20)
plt.xlabel('Mean Value', fontsize=12)
plt.ylabel('Features', fontsize=12)

for i, v in enumerate(mean_values.values):
    ax.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig("Figure Mean Values After SMOTE.pdf", dpi=100)
plt.show()

plt.figure(figsize=(10, 6))
std_values = X_resampled.std().sort_values(ascending=False)
ax = sns.barplot(x=std_values.values, y=std_values.index, palette="plasma")
plt.title('Standard Deviation of Features After SMOTE', fontsize=16, pad=20)
plt.xlabel('Standard Deviation', fontsize=12)
plt.ylabel('Features', fontsize=12)

for i, v in enumerate(std_values.values):
    ax.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig("Figure Standard Deviation After SMOTE.pdf", dpi=100)
plt.show()

# ---------------- T-SNE VISUALIZATION ----------------
print("\nRunning t-SNE... (may take some time)")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_embedded = tsne.fit_transform(X_resampled)

tsne_df = pd.DataFrame({
    'TSNE1': X_embedded[:,0],
    'TSNE2': X_embedded[:,1],
    'Class': y_resampled
})

plt.figure(figsize=(8,6))
sns.scatterplot(
    x='TSNE1', y='TSNE2',
    hue='Class',
    palette='tab10',
    data=tsne_df,
    alpha=0.7
)
plt.title("t-SNE Visualization of Resampled Data")
plt.legend(title="Class")
plt.tight_layout()
plt.draw()
plt.savefig("Figure t-SNE Visualization of Resampled Data.pdf", dpi=100)
plt.show()
