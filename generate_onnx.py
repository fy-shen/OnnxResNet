import torch
import torchvision.models as models


h, w = 224, 224
onnxFile = 'data/resnet18.onnx'
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights, progress=False).eval().cuda()

torch.onnx.export(
    model,
    torch.randn(1, 3, h, w, device='cuda'),
    onnxFile,
    input_names=['x'],
    output_names=['y'],
    do_constant_folding=True,
    verbose=True,
    keep_initializers_as_inputs=True,
    opset_version=12,
    dynamic_axes={'x': {0: 'nBatchSize'}, 'y': {0: 'nBatchSize'}}
)
