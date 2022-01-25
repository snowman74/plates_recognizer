# plates_recognizer
yolov5 models with fastapi web backend


Проект по распознаванию автомобильных номеров.

В качестве системы распознавания использованы 2 модели YOLO v5, (https://github.com/ultralytics/yolov5), обученные на примерно 10 000 фотографиях.

В качестве backend применен FastAPI, также в качестве демонстрации присутствует HTML страница.

Проект обёрнут в Docker образ, который Вы можете найти по адресу: https://hub.docker.com/repository/docker/snowman74/plates_recognizer

______________________________________________________________________________________________________________

Для запуска можете скопировать данный репозиторий, либо воспользоваться ссылкой на Docker образ выше.

Шаги для запуска:

    git clone https://github.com/snowman74/plates_recognizer.git
    cd plates_recognizer/
    uvicovn main:app --reload
    
    backend FastApi по умолчанию запускается по адресу:
    http://127.0.0.1:8000/docs#/
    
    страничку с демонстрацией можно посмотреть по адресу:
    http://127.0.0.1:8000/monolith/home/
    

