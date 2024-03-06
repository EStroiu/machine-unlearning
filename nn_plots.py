# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import tensorflow as tf
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import roc_auc_score, roc_curve
# import json

# energy_anomaly = pd.read_csv('energy_anomaly_detection/train_features.csv')
# energy_anomaly = energy_anomaly.dropna(subset=['meter_reading'])

# X = energy_anomaly[['meter_reading', 'month', 'square_feet', 'weekday', 'hour']]
# y = energy_anomaly['anomaly']

# X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_val_scaled = scaler.transform(X_val)
# X_test_scaled = scaler.transform(X_test)

# def create_model(hidden_layers=1, units=32, activation='relu'):
#     model = tf.keras.Sequential()
#     model.add(tf.keras.layers.Dense(units, activation=activation, input_shape=(X_train.shape[1],)))
#     for _ in range(hidden_layers - 1):
#         model.add(tf.keras.layers.Dense(units, activation=activation))
#     model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
#     model.compile(optimizer='adam',
#                   loss='binary_crossentropy',
#                   metrics=[tf.keras.metrics.AUC()])
#     return model

# param_grid = {
#     'hidden_layers': [1, 2, 3],
#     'units': [16, 32, 64, 128, 256],
#     'activation': ['relu', 'tanh', 'sigmoid', 'linear']
# }

# keras_classifier = tf.keras.wrappers.scikit_learn.KerasClassifier(create_model)

# random_search = RandomizedSearchCV(keras_classifier, param_distributions=param_grid, n_iter=10, cv=2)
# random_search_result = random_search.fit(X_train_scaled, y_train)

# # Save best hyperparameters to a JSON file
# best_params = random_search_result.best_params_
# with open('best_hyperparameters.json', 'w') as f:
#     json.dump(best_params, f)

# best_model = random_search_result.best_estimator_.build_fn()
# best_model.fit(X_train_scaled, y_train, epochs=5, batch_size=32, validation_data=(X_val_scaled, y_val))

# epoch_stats = pd.DataFrame(best_model.history.history)
# epoch_stats.to_csv('plots/epoch_statistics.csv', index=False)

# y_pred_proba = best_model.predict(X_test_scaled)
# auc_roc = roc_auc_score(y_test, y_pred_proba)
# print('AUC-ROC:', auc_roc)

# results_df = pd.DataFrame({
#     'True Labels': y_test,
#     'Predictions': y_pred_proba.flatten()
# })
# results_df.to_csv('plots/results.csv', index=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

energy_anomaly = pd.read_csv('energy_anomaly_detection/train_features.csv')
energy_anomaly = energy_anomaly.dropna(subset=['meter_reading'])

X = energy_anomaly[['meter_reading', 'month', 'square_feet', 'weekday', 'hour']]
y = energy_anomaly['anomaly']

# building_hours = energy_anomaly['building_hour'].str.split('-').str[0].astype(int)
# X['building_hour'] = building_hours

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.AUC()])

history = model.fit(X_train_scaled, y_train, epochs=30, batch_size=32, validation_data=(X_val_scaled, y_val))

epoch_stats = pd.DataFrame(history.history)
epoch_stats.to_csv('epoch_statistics.csv', index=False)

y_pred_proba = model.predict(X_test_scaled)
auc_roc = roc_auc_score(y_test, y_pred_proba)
print('AUC-ROC:', auc_roc)

results_df = pd.DataFrame({
    'True Labels': y_test,
    'Predictions': y_pred_proba.flatten()
})

results_df.to_csv('results.csv', index=False)

# first attempt  [tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)), tf.keras.layers.Dense(1, activation='sigmoid')]
