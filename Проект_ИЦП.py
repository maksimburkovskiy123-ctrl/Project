import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree

df = pd.read_csv("Spotify_final_dataset.csv")
# Проверка, что данные считались хорошо
print(df.info)
print(df.dtypes)
print(df.columns.tolist())
print(df.isna().sum()) # смотрим сколько пропусков, и где они находятся
df = df.dropna() # удалили строчки, где есть пропуск   
print(df.describe())


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

ft = ['Position', 'Days', 'Peak Position (xTimes)', 'Total Streams', 
      'Artist Name encoded', 'Peak Streams', 'Song Name encoded']
scaler = StandardScaler()
df_scaled=df.copy()
df_scaled[ft] = scaler.fit_transform(df[ft])

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


df['Hit'] = ((df['Peak Position'] <= 20) & (df['Top 10 (xTimes)'] > df['Top 10 (xTimes)'].median())).astype(int)
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
