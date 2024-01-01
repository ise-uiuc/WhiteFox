
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.MaxPool2d(3, stride=1, padding=1, dilation=1, return_indices=False, ceil_mode=False)
    def forward(self, x1):
        l1 = self.avgpool(x1)
        return l1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
