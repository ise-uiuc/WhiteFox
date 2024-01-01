
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = F.conv_transpose3d(x1, torch.randn(3, 3, 3, 3, 3), bias=None, stride=2, padding=0, dilation=2)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 4, 4)
