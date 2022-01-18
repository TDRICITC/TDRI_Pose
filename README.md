
# How to build image
```


## build docker image
image_name="tdri/yolov5-python-flask:01"
docker build -t "$image_name" .

```
# Run Container 
### run as deamon 
```
container_name="yolov5-flask-01"
image_name="tdri/yolov5-python-flask:01"
docker run -d all --name ${container_name} \
    -p 5000:5000 \
    ${image_name} \
    tail -f /dev/null
```
### direct run flask(GPU) 
```
container_name="yolov5-flask-01"
image_name="tdri/yolov5-python-flask:01"
docker run -d --gpus all --restart=always --name ${container_name} \
    -p 5000:5000 \
    ${image_name} \
    gunicorn -w 20 -b :5000 flask_app:app

```
### direct run flask(CPU) 
```
container_name="yolov5-flask-01"
image_name="tdri/yolov5-python-flask:01"
docker run -d --restart=always --name ${container_name} \
    -p 5000:5000 \
    ${image_name} \
    gunicorn -w 20 -b :5000 flask_app:app ##平行處理設定Thread = 20

```
