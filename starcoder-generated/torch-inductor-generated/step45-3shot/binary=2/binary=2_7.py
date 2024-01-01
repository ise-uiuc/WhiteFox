
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.conv2d(x1,weight=torch.zeros([1,1,3,3]), bias=None, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1)
        v2 = v1 - 1.0
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
