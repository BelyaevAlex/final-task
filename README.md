# final-task
# Как установить и настроить
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
    Путь к датасету
    ```
    -d: Path
    ```
    Penalty: l2 или none
    ```
    -p: str
    ```
    max_iter
    ```
    -m: int
    ```
    Путь для сохранения модели
    ```
    -s: Path
    ```
    Использование скалирования
    ```
    -u: bool
    ```
    разделение датасета
    если True -> тройное разделение
    если False -> тройное разделение
    ```
    -sd: bool
    ```
    размер тестового датасета
    ```
    -t: float (0 -> 1)
    ```
    выбор модели
    если True -> logistic regression
    если False -> K neighbor classifier
    ```
    -lg: bool
    ```
    n_neighbors
    ```
    -n: int(odd)
    ```
    использование pca
    ```
    -pca: bool
    ```
    использование grid search
    ```
    -gs: bool
    ```
![изображение](https://user-images.githubusercontent.com/77803344/166120675-d7a4f1d5-cee8-4e53-ad7f-deb57c6164ef.png)
![изображение](https://user-images.githubusercontent.com/77803344/166200839-b533f927-1ada-4590-976b-f1fe14a0a361.png)
![изображение](https://user-images.githubusercontent.com/77803344/166311770-129c5d46-7e59-4eae-ba7d-98f1670ce4eb.png)

