from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Abell 2218
ra = 16 + (35 / 60) + (51.89 / 3600)
dec = 66 + (12 / 60) + (38.71 / 3600)
radius = 1 * u.deg

coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')


# In here I choose the rows I want to examine in df from the Gaia dataset(SQL query)
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
job = Gaia.launch_job_async(f"""
    SELECT source_id, ra, dec, parallax, pmra, pmdec, phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {coord.ra.deg}, {coord.dec.deg}, {radius.to(u.deg).value})
    ) = 1
""")


query_sql = job.get_results()
df = query_sql.to_pandas()

print("First 5 row:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nShape:")
print(df.shape)

print("\nNumber of Missing Data:")
print(df.isnull().sum())


print("\nBasic Statistics:")
print(df.drop(columns=['SOURCE_ID', 'ra', 'dec']).describe())




from mlxtend.plotting import scatterplotmatrix

features = ['parallax', 'pmra', 'pmdec', 'phot_g_mean_mag']
df_selected = df[features]

fig, axes = scatterplotmatrix(df_selected.values,
                              figsize=(10, 8),
                              alpha=0.5,
                              names=features)

plt.tight_layout()
plt.show()



X = df.drop(columns=['SOURCE_ID', 'ra', 'dec'])
y = X['parallax']
# Aim to use parallax row.

#Shuffling
indices = np.arange(X.shape[0])
rng = np.random.RandomState(123)
permuted_indices = rng.permutation(indices)

# Train, Validation and Test subsets
train_size = int(0.65 * X.shape[0])
valid_size = int(0.15 * X.shape[0])
test_size = X.shape[0] - (train_size + valid_size)

print(f"Train size: {train_size}, Validation size: {valid_size}, Test size: {test_size}")

train_ind = permuted_indices[:train_size]
valid_ind = permuted_indices[train_size:(train_size + valid_size)]
test_ind = permuted_indices[(train_size + valid_size):]

X_train, y_train = X.iloc[train_ind], y.iloc[train_ind]
X_valid, y_valid = X.iloc[valid_ind], y.iloc[valid_ind]
X_test, y_test = X.iloc[test_ind], y.iloc[test_ind]

print(f"X_train shape: {X_train.shape}")




X = X.fillna(X.mean())
y = y.fillna(y.mean())

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.neighbors import KNeighborsRegressor

knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X_train, y_train)


y_pred = knn_model.predict(X_valid)
y_pred = knn_model.predict(X_valid)

print("Predicted Parallax Values for the Validation Set:")
print(y_pred)


from sklearn.metrics import mean_squared_error

X_test = X_test.fillna(X_test.mean())
y_test_pred = knn_model.predict(X_test)

plt.figure(figsize=(8, 6))

plt.scatter(y_valid, y_pred, color='blue', label='Validation Set', alpha=0.5)

plt.scatter(y_test, y_test_pred, color='red', label='Test Set', alpha=0.5)

plt.plot([y.min(), y.max()], [y.min(), y.max()], color='black', linestyle='--')

plt.xlabel('True Parallax')
plt.ylabel('Predicted Parallax')
plt.title('True vs Predicted Parallax Values')
plt.legend()
plt.show()


print(f"NaN values in X: {X_test.isna().sum().sum()}")


print(f"NaN values in X: {y_test.isna().sum().sum()}")


from sklearn.model_selection import train_test_split

X = X.fillna(X.mean())
y = y.fillna(y.mean())

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.35,
                                                    shuffle=True, random_state=42, stratify=y)

X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.43,
                                                    shuffle=True, random_state=42, stratify=y_temp)



print(f"Class proportions in train set: {np.bincount(y_train)}")
print(f"Class proportions in validation set: {np.bincount(y_valid)}")
print(f"Class proportions in test set: {np.bincount(y_test)}")


print(f"Class proportions: {np.bincount(y)}")


from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Sayısal verilerimizi seçelim
numeric_columns = ['parallax', 'pmra', 'pmdec', 'phot_g_mean_mag']
df_numeric = df[numeric_columns]


# Min-Max normalization
scaler_min_max = MinMaxScaler()
df_numeric_min_max = scaler_min_max.fit_transform(df_numeric)
df_numeric_min_max = pd.DataFrame(df_numeric_min_max, columns=df_numeric.columns)

print("Min-Max Normalizasyonu Uygulandı:\n", df_numeric_min_max.head())


# Z-Score standardization
scaler_standard = StandardScaler()
df_numeric_standard = scaler_standard.fit_transform(df_numeric)
df_numeric_standard = pd.DataFrame(df_numeric_standard, columns=df_numeric.columns)

print("\nZ-Score Standardizasyonu Uygulandı:\n", df_numeric_standard.head())


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_valid_std = scaler.transform(X_valid)
X_test_std = scaler.transform(X_test)

# **Categorical Data**


print("\nData Types:")
print(df.dtypes)


# df['distance_pc'] = df['parallax'].apply(lambda p: 1000 / p if p > 0 else None)

# print(df[['parallax', 'distance_pc']].head())


df['coords'] = df.apply(
    lambda row: f"{row['ra']}° {row['dec']}°", axis=1
)

print(df[['ra', 'dec', 'coords']].head())

print(df.columns)

from sklearn.impute import SimpleImputer
import pandas as pd

df_numeric = df.drop(columns=['coords'])  # Drop the 'coords' column

imputer = SimpleImputer(strategy='mean')

X = df_numeric.values
X_imputed = imputer.fit_transform(X)
# df_imputed = pd.DataFrame(X_imputed, columns=df_numeric.columns)

# print(df_imputed.head())


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

## I wrote the code in bulk from at the beginning

X = df.drop(columns=['SOURCE_ID', 'ra', 'dec', 'coords'])
y = X['parallax']
X = X.fillna(X.mean())
y = y.fillna(y.mean())


indices = np.arange(X.shape[0])
rng = np.random.RandomState(123)
permuted_indices = rng.permutation(indices)

train_size = int(0.65 * X.shape[0])
valid_size = int(0.15 * X.shape[0])
test_size = X.shape[0] - (train_size + valid_size)

train_ind = permuted_indices[:train_size]
valid_ind = permuted_indices[train_size:(train_size + valid_size)]
test_ind = permuted_indices[(train_size + valid_size):]

X_train, y_train = X.iloc[train_ind], y.iloc[train_ind]
X_valid, y_valid = X.iloc[valid_ind], y.iloc[valid_ind]
X_test, y_test = X.iloc[test_ind], y.iloc[test_ind]

pipe = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=3))

pipe.fit(X_train, y_train)

predictions = pipe.predict(X_test)

print("Predictions:", predictions)

pipe


pipe.fit(X_train, y_train)
pipe.predict(X_test)


from sklearn.model_selection import GridSearchCV
from mlxtend.evaluate import PredefinedHoldoutSplit
from sklearn.pipeline import make_pipeline

params = {'kneighborsregressor__n_neighbors': [1, 3, 5],
          'kneighborsregressor__p': [1, 2]}

split = PredefinedHoldoutSplit(valid_indices=valid_ind)

grid = GridSearchCV(pipe,
                    param_grid=params,
                    cv=split)

grid.fit(X, y)

grid.cv_results_
print("Best score:", grid.best_score_)
print("Best parameters:", grid.best_params_)
clf = grid.best_estimator_
clf.fit(X_train, y_train)
print('Test accuracy: %.2f%%' % (clf.score(X_test, y_test)*100))