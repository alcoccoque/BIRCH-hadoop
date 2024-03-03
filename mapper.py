#!/usr/bin/env python

import sys
from sklearn.cluster import Birch

# Ініціалізація та навчання моделі BIRCH
birch = Birch(n_clusters=3)  # Припустимо, що кількість кластерів - 3

# Створення списку для зберігання даних
data = []

# Зчитування вхідних даних з Hadoop Streaming
for line in sys.stdin:
    # Розділити рядок на ключ і значення
    key, value = line.strip().split('\t')

    # Перетворення рядка на потрібні дані для кластеризації
    data_point = [float(coord) for coord in value.split(',')]

    # Додавання даного до списку даних
    data.append(data_point)

# Навчання моделі BIRCH
birch.fit(data)

# Виведення результатів кластеризації
for point in data:
    # Вибір кластера для даної точки
    cluster_id = birch.predict([point])[0]

    # Виведення кластера та точки
    print(f"{cluster_id}\t{','.join(str(coord) for coord in point)}")
