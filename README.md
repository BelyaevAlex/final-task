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
    default="./data/train.csv"
    ```
    Penalty: l2 или none
    ```
    -p: str
    default="l2"
    ```
    max_iter
    ```
    -m: int
    default=100
    ```
    Путь для сохранения модели
    ```
    -s: Path
    default="./data/model.joblib"
    ```
    Использование скалирования
    ```
    -u: bool
    default=True
    ```
    разделение датасета
    если True -> двойное разделение
    если False -> тройное разделение
    ```
    -sd: bool
    default=True
    ```
    размер тестового датасета
    ```
    -t: float (0 -> 1)
    default=0.2
    ```
    выбор модели
    если True -> logistic regression
    если False -> K neighbor classifier
    ```
    -lg: bool
    default=True
    ```
    n_neighbors
    ```
    -n: int(odd)
    default=5
    ```
    использование pca
    ```
    -pca: bool
    default=False
    ```
    использование grid search
    ```
    -gs: bool
    default=False
    ```
    использование pandas_profiling
    ```
    -prof: bool
    default=True
    ```
#Почему я использую src layout
  Из-за его удобства в использовании, лёгкого понимания и из-за того что он используется в примере
  ![изображение](https://user-images.githubusercontent.com/77803344/166120675-d7a4f1d5-cee8-4e53-ad7f-deb57c6164ef.png)
  ![изображение](https://user-images.githubusercontent.com/77803344/166200839-b533f927-1ada-4590-976b-f1fe14a0a361.png)
  ![изображение](https://user-images.githubusercontent.com/77803344/166504275-da57349b-0bd1-4d04-9cec-13c50d50ce55.png)
  ![изображение](https://user-images.githubusercontent.com/77803344/166739098-b1ff09a7-61a5-418f-bae3-415af92e0d79.png)

  ![изображение](https://user-images.githubusercontent.com/77803344/168489265-1451d297-0977-489e-980d-c3846532e5c6.png)
![изображение](https://user-images.githubusercontent.com/77803344/168489531-7fd0fbdb-55d8-436e-a779-a9a244f28e62.png)

