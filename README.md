# Предсказание веса и габарита товаров по метаданным о товаре

## Описание

Задача – научиться по данным о товаре, указанным в объявлении, предсказывать четыре величины:
- weight - вес (кг),
- height - высота (см),
- width - ширина (см),
- length - длина (см).

Для этого были обучены эмбеддинги названий и описаний, изображений, которые затем были
конкатенированы с табличными данными для получения эмбеддингов всего объявления.

На этих эмбеддингах обучены линейные модели, случайный лес и градиентный бустинг:
baseline_item_embeddings.ipynb

Лучший результат показала простая нейросеть:
final_item_embeddings.ipynb

В дополнительных ноутбуках описаны: 
- EDA - eda.ipynb;
- extracting embeddings from texts - texts_analysis.ipynb;
- extracting embeddings from images - images_embedder.ipynb

И вспомогательные классы и функции в файлах в директории preprocessing

## Требования

- Python 3.10+
- requirements.txt

## Установка и запуск

1. Клонировать или скачать репозиторий:
   ```bash
   git clone https://github.com/<ссылка_на_репозиторий>.git
   cd <папка_с_проектом>
   ```

2. Установить все необходимые библиотеки согласно requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Скачать эмбеддинги и веса модели:
   ```
   python download_files.py
   ```

4. Запустить ноутбук(и):
   ```
   run all(Ctr+Alt+Shift+Enter) 
   ```


## Структура проекта

```
"""
!!! Для загрузки эмбеддингов(директория image_embeddings) нужно запустить preprocessing/download_files.py
"""

final_project
└───README.md                          # Описание проекта, структура, требования 
└───eda.ipynb                          # EDA и baseline модели
└───images_embedder.ipynb              # Модель для получения эмбеддингов изображений
└───texts_analysis.ipynb               # Анализ текстов
└───baseline_item_embeddings.ipynb     # Baseline on item embeddings
└───final_item_embeddings.ipynb        # Best result on item embeddings (MLP)
└───metrics.py                         # Метрики
└───requirements.txt                   # requirements
└───preprocessing   
    └─── download_files.py             # Скрипт для скачивания эмбеддингов изображений и весов модели
    └─── metadata_preprocessing.py     # Весь препроцессинг данных
    └─── text_preprocessing.py         # Класс и функции для токенизации и кодирования названий и описаний объявлений
└───image_embeddings   
    └─── train_embeddings.parquet      # embeddings of train images
    └─── test_embeddings.parquet       # embeddings of test images
    └─── efficientnet_b6_weights.pth   # weights of model for encoder
└───data   
    └─── input      
         └─── test.parquet             # test dataset
         └─── test.parquet             # train dataset
    └─── output
         └─── results.txt              # Использованные модели и результаты метрики на тесте
         └─── ...                      # csv с предсказаниями моделей
    └─── indexes_to_ignore.csv         # ids данных с выбросами в train
```