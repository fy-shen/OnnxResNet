
ResNet-18 to ONNX
```shell
python generate_onnx.py
```

```
mkdir build
cd build
cmake ..
make
cd ../bin

./demo
./demo --fp16
./demo --int8
```