# ClothSegmentNCNN-libhv

## Input

- `<hostname>:<port>/segment`
- `POST`
- (Default) `8081` port
- `form-data`
- body: `image` as `File` type

## Output

- `form-data`
- boundary: `----WebKitFormBoundary7MA4YWxkTrZu0gW`
- result: (json)
  - count
  - latency (ms)
  - objects (list)
    - label
    - prob
    - rect (json)
      - x
      - y
      - width
      - height

**Example**

```text
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="result"
Content-Type: application/json

{"count":2,"latency":114,"objects":[{"label":short sleeve top,"prob":0.914738,"rect":{"x":146.758530,"y":48.110760,"width":951.941650,"height":1134.188110}},{"label":shorts,"prob":0.902601,"rect":{"x":84.707527,"y":767.112671,"width":749.657532,"height":539.034912}}]}
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="segment_0.png"
Content-Type: image/jpeg
Content-Length: 2037725

�PNG
...
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="segment_1.png"
Content-Type: image/jpeg
Content-Length: 1927837

�PNG
...
------WebKitFormBoundary7MA4YWxkTrZu0gW--
```
## Environment

- Ubuntu 22.04

## Dependence

- gcc 11.4.0
- GNU make 4.3
- CMake 3.22.1
- libhv
- OpenCV
- NCNN

## Install from source

### Install package

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install make camke gcc wget -y
```

### Install OpenCV

1. Download source code.

   ```bash
   wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
   unzip opencv.zip
   mv opencv-4.x opencv
   ```

2. Configure and build.

   ```bash
   mkdir -p build && cd build
   cmake ..
   make -j `nproc`
   ```

3. Install.

   ```bash
   sudo make install
   ```

### Install NCNN

1. Download NCNN release from [Tencent/ncnn: ncnn is a high-performance neural network inference framework optimized for the mobile platform (github.com)](https://github.com/Tencent/ncnn) 
2. unzip it to `<workspace>/ClothSegmentNCNN-libhv/lib/`

### Install libhv

1. Download libhv source code from [ithewei/libhv: 🔥 比libevent/libuv/asio更易用的网络库。A c/c++ network library for developing TCP/UDP/SSL/HTTP/WebSocket/MQTT client/server. (github.com)](https://github.com/ithewei/libhv/tree/master) 

2. Configure and build.

   ```bash
   mkdir -p build && cd build
   cmake ..
   make -j `nproc`
   ```

3. Install.

   ```bash
   sudo make install
   ```

### Build project source code

1. Pull source code

   ```bash
   git pull https://github.com/FaterYU/ClothSegmentNCNN-libhv.git
   cd ClothSegmentNCNN-libhv
   ```

2. Configure and build

   ```bash
   mkdir -p build && cd build
   cmake ..
   make -j `nproc`
   cd ..
   ```

### Launch executable program

```bash
./bin/yolo8_ncnn
```

## Install from Docker

[fateryu/wearwizard - Docker Image | Docker Hub](https://hub.docker.com/r/fateryu/wearwizard)

### Install Docker

```bash
curl -fsSL https://test.docker.com -o test-docker.sh
sudo sh test-docker.sh
```

## Pull image and run

```bash
docker run -itd --name wear_seg \
--privileged --network host --restart always \
fateryu/wearwizard:segment \
/yolo8_ncnn
```
