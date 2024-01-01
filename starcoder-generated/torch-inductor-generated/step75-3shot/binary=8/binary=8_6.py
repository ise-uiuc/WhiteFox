
# model from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = F.adaptive_avg_pool2d(x1, [10, 10])
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
