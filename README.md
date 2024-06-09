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
