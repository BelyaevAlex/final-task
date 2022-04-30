# final-task
1. Как установить и настроить
  1. Скачайте этот репозиторий к себе на компьютер
  2. Убедитесь, что на вашем компьютере установлены Python 3.9 и [Poetry] (https://python-poetry.org/docs/) (я использую Poetry 1.1.13)
  3. Установите зависимости проекта (*запустите эту и следующие команды в терминале из корня клонированного репозитория*):
  ```sh
  poetry install --no-dev
  ```
  4. Запустите train соответствующей командой:
  ```sh
  poetry run train
  ```
  5. Запустите пользовательский интерфейс MLflow, чтобы просмотреть информацию о проведенных вами экспериментах:
  ```sh
  poetry run mlflow ui
  ```
2. Настройки для train:
  1. 
    -d: str
    
    Путь к датасету
  2. -p: str
    Penalty: l2 или none
  3. -m
    max_iter
  4. -s
    Путь для сохранения модели
  5. -u
    Использование скалирования
  6. -sd
    разделение датасета
    если True -> тройное разделение
    если False -> тройное разделение
  7. -t
    размер тестового датасета
  8. -lg 
    выбор модели
    если True -> logistic regression
    если False -> K neighbor classifier
  9. -n 
    n_neighbors
  10. -pca
    использование pca
  11. -gs
    использование grid search
  
