import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("Spotify_final_dataset.csv")
'''
# Проверка, что данные считались хорошо
print(df.info)
print(df.dtypes)
print(df.columns.tolist())
print(df.isna().sum()) # смотрим сколько пропусков, и где они находятся
'''

df = df.dropna() # удалили строчки, где есть пропуск   
#print(df.describe())

'''
# Диаграмма
top_artist = df['Artist Name'].value_counts().head(14) # 14, а не 15, потому что 15 - гении маркетинга, но я их не признаю
plt.figure(figsize=(11,7))
plt.bar(top_artist.index, top_artist.values)
plt.title('Топ 14 исполнителей в Spotify по хитам')
plt.xlabel('Исполнитель')
plt.ylabel('Количество хитовых треков')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#Гистограмма
fig, axs = plt.subplots(1, 2, figsize=(13, 7))
axs[0].hist(df['Total Streams'], bins=50, color = 'gold', edgecolor='black', alpha =0.7)
axs[0].set_title('Гистограмма для Total Streams')
axs[0].set_xlabel('Всего прослушиваний')
axs[0].set_ylabel('Частота')

axs[1].hist(df['Peak Streams'], bins=50, color = 'green', edgecolor='black', alpha =0.7)
axs[1].set_title('Гистограмма для Peak Streams')
axs[1].set_xlabel('Прослушивания за день')
axs[1].set_ylabel('Частота')
plt.tight_layout()
plt.show()


# Boxplot - смотрим выбросы 
cr = ['Total Streams', 'Peak Streams', 'Days']
boxplot = [df[col] for col in cr]   
plt.figure(figsize=(13,7))
plt.boxplot(boxplot, labels=cr)
plt.title('Boxplot для просмотра выбрасов')
plt.tight_layout()
plt.show()
'''

'''
Можно заметить, что очень много выбросов. Выбросы - самые популярные треки, сами коробки - обычные треки.
Учитывая, что это данные песен Spotify, то такие выбросы будут нормой

Total Streams - важный признак, так как показывает популярность трека
Peak Streams - важный признак, так как с его помощью можно посмотреть насколько сильно трек завирусился
Peak Position (xTimes) - признак, который требует очисти (Должно остаться только числовое значение)
Было 4 пропуска в Sog Name, можно было заменить, но так как их мало я удалил
'''

df['Peak Position (xTimes)'] = df['Peak Position (xTimes)'].str.replace('(x', '').str.replace(')', '').astype(int)
# Произвели очистку столбца Peak Position (xTimes)

Encoder_artist = LabelEncoder()
Encoder_song = LabelEncoder()
df['Artist Name encoded'] = Encoder_artist.fit_transform(df['Artist Name'])
df['Song Name encoded'] = Encoder_song.fit_transform(df['Song Name'])
#Закодировали строки, чтобы sklearn мог работать дальше

df['Hit'] = ((df['Peak Position'] <= 20) & (df['Top 10 (xTimes)'] > df['Top 10 (xTimes)'].median())).astype(int)

ft = ['Position', 'Days', 'Peak Position (xTimes)', 'Total Streams', 
      'Artist Name encoded', 'Peak Streams', 'Song Name encoded']

'''
target = df['Hit']
base_model = LogisticRegression()

X_raw = df[ft]
X_train, X_test, y_train, y_test = train_test_split(X_raw, target, test_size=0.2, random_state=35)
base_model.fit(X_train, y_train)
acc_raw = accuracy_score(y_test, base_model.predict(X_test))
print("\n Accuracy сырых данных:", acc_raw)

scaler_std = StandardScaler()
X_std = scaler_std.fit_transform(df[ft])

X_train, X_test, y_train, y_test = train_test_split(X_std, target, test_size=0.2, random_state=35)
base_model.fit(X_train, y_train)
acc_std = accuracy_score(y_test, base_model.predict(X_test))
print("Accuracy StandardScaler:", acc_std)

scaler_mm = MinMaxScaler()
X_mm = scaler_mm.fit_transform(df[ft])
X_train, X_test, y_train, y_test = train_test_split(X_mm, target, test_size=0.2, random_state=35)
base_model.fit(X_train, y_train)
acc_mm = accuracy_score(y_test, base_model.predict(X_test))
print("Accuracy (MinMaxScaler):", acc_mm)

df_clip = df.copy()
for col in ['Total Streams', 'Peak Streams']:
    q_low = df_clip[col].quantile(0.05)
    q_high = df_clip[col].quantile(0.95)
    df_clip[col] = df_clip[col].clip(q_low, q_high)

X_clip = df_clip[ft]
X_clip_std = StandardScaler().fit_transform(X_clip)
X_train, X_test, y_train, y_test = train_test_split(X_clip_std, target, test_size=0.2, random_state=35)
base_model.fit(X_train, y_train)
acc_clip = accuracy_score(y_test, base_model.predict(X_test))
print("Accuracy (Обрезка выбросов + StandardScaler):", acc_clip)

df_log = df.copy()
df_log['Total Streams'] = np.log1p(df_log['Total Streams'])
df_log['Peak Streams'] = np.log1p(df_log['Peak Streams'])
X_log = df_log[ft]
X_log_std = StandardScaler().fit_transform(X_log)
X_train, X_test, y_train, y_test = train_test_split(X_log_std, target, test_size=0.2, random_state=35)
base_model.fit(X_train, y_train)
acc_log = accuracy_score(y_test, base_model.predict(X_test))
print("Accuracy (Log + StandardScaler):", acc_log)

results_preprocessing = pd.DataFrame({
      'Preprocessing': ['Сырые данные','StandardScaler','MinMaxScaler','Обрезание выбросов + StandardScaler','Log + StandardScaler'],
      'Accuracy': [acc_raw, acc_std, acc_mm, acc_clip, acc_log]
      })

print(results_preprocessing)

plt.figure(figsize=(10, 5))
plt.bar(results_preprocessing['Preprocessing'], results_preprocessing['Accuracy'])
plt.xticks(rotation=45, ha='right')
plt.ylabel("Accuracy")
plt.title("Влияние предобработки на качество Logistic Regression")
plt.tight_layout()
plt.savefig("Сравнение.png")
plt.close()
'''








scaler = StandardScaler()
df_scaled=df.copy()
df_scaled[ft] = scaler.fit_transform(df[ft])

'''
fig, axs = plt.subplots(1, 2, figsize=(13, 7))
axs[0].hist(df_scaled['Total Streams'], bins=50, color='pink', edgecolor='black', alpha =0.7)
axs[0].set_title('Стандартизированная гистограмма для Total Streams')
axs[0].set_xlabel('Всего прослушиваний')
axs[0].set_ylabel('Частота')

axs[1].hist(df_scaled['Peak Streams'], bins=50, color='red', edgecolor='black', alpha =0.7)
axs[1].set_title('Стандартизированная гистограмма для Peak Streams')
axs[1].set_xlabel('Прослушивания за день')
axs[1].set_ylabel('Частота')
plt.tight_layout()
plt.show()

cr = ['Total Streams', 'Peak Streams', 'Days']
boxplot = [df_scaled[col] for col in cr]   
plt.figure(figsize=(13,7))
plt.boxplot(boxplot, labels=cr)
plt.title('Стандатизированный Boxplot для просмотра выбрасов')
plt.tight_layout()
plt.show()
'''


# Столкнулся с проблемой accuracy 1.0. Понял, что у Hit это те же признаки, что и в модели обучения
# После их удаления получил хороший accuracy
x_scaled = df_scaled[ft]
target = df['Hit']
# Обучающая и тестовая выборка
X_train, X_test, y_train, y_test = train_test_split(x_scaled, target, test_size=0.2, random_state=35)

dt = DecisionTreeClassifier(random_state=35)
knn = KNeighborsClassifier(n_neighbors=5)
logreg = LogisticRegression()
dt.fit(X_train, y_train)
knn.fit(X_train, y_train)
logreg.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
y_pred_knn = knn.predict(X_test)
y_pred_logreg = logreg.predict(X_test)

'''
cm = confusion_matrix(y_test, y_pred_logreg)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

plt.subplots(1,1,figsize = (10,10))
tree.plot_tree(dt, filled = True)
plt.show()


accuracy_dt = accuracy_score(y_test, y_pred_dt)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)

print("Accuracy for dt:", accuracy_dt)
print("Accuracy for knn:", accuracy_knn)
print("Accuracy for logreg:", accuracy_logreg)

print(confusion_matrix(y_test, y_pred_logreg))
print(classification_report(y_test, y_pred_logreg))
'''
'''
def regression_metrics(y_true, y_pred):  
    mae = mean_absolute_error(y_true, y_pred)  # MAE показывает, на сколько в среднем модель ошибается
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) # RMSE в отличии от mae более чувствительна к большим ошибкам 
    r2 = r2_score(y_true, y_pred) # R^2 показывает, насколько хорошо модель объясняет данные в дополнение к rmse
    return mae, rmse, r2

reg_target_col = 'Total Streams'
ft_reg = [c for c in ft if c != reg_target_col]

X_reg = df_scaled[ft_reg]
y_reg = df_scaled[reg_target_col]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=35)
reg_models = {
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(random_state=35),
    "KNNRegressor (k=5)": KNeighborsRegressor(n_neighbors=5)
}
reg_results = []

for name, model in reg_models.items():
    model.fit(X_train_r, y_train_r)
    pred = model.predict(X_test_r)

    mae, rmse, r2 = regression_metrics(y_test_r, pred)

    reg_results.append({
        "Model": name,
        "Target": reg_target_col,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })
    
reg_results_df = pd.DataFrame(reg_results).sort_values(by="RMSE") # модель с меньшим RMSE делает меньше крупных ошибок, важно для прогноза популярности
print(reg_results_df)
best_model_name = reg_results_df.iloc[0]["Model"]
best_model = reg_models[best_model_name]
best_model.fit(X_train_r, y_train_r)
best_pred = best_model.predict(X_test_r) 
print("Первые 10 реальных значений:", np.round(y_test_r.values[:10], 4))
print("Первые 10 предсказанных значений:", np.round(best_pred[:10], 4))
'''

knn_accuracy = []
dt_accuracy = []
for k in range(1,21):
    knn = KNeighborsClassifier(n_neighbors=k)
    dt = DecisionTreeClassifier(max_depth=k, random_state=35)
    knn.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    y_pred_dt = dt.predict(X_test)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    acc_dt = accuracy_score(y_test, y_pred_dt)
    knn_accuracy.append(acc_knn)
    dt_accuracy.append(acc_dt)

res = pd.DataFrame({
    "k": range(1, 21), 
    "Accuracy_knn": knn_accuracy,
    "max_depth": range(1,21),
    "Accuracy_dt": dt_accuracy
})

print(res) # k = 8 для knn, k = 4 для dt

