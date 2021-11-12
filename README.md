# emP:ty main server

일정 시간마다 주차장에 설치된 라즈베리 파이 카메라로부터 주차장 이미지를 가져오고, detectron2를 이용하여 주차장 full/empty 여부를 제공해주는 서버.

### before start
1. detectron2 설치

    [m1 설치 방법](https://velog.io/@huttzza/m1-detectron2-%EC%84%A4%EC%B9%98)

2. flask 설치

    ```
    pip install flask
    pip install flask-cors
    ```

### run
1. set constant.py
    ```
    PI_IP = rasberry pi ip
    CFG_FILE = local cfg path
    ```
    (set `static` folder too)

2. run

    ```
    sudo python3 main.py
    ```

### for detect
to local host (POST)
```
{
    "mode" : "local" or "url",
    "imageUrl" : imageUrl or empty
}
```
