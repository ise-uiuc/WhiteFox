
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(2, stride=2, dilation=2, return_indices=True)
    def forward(self, x1):
        v1 = x1
        v2, _ = self.maxpool(v1)
        v3 = v2.permute(0, 2, 3, 1)
        return v3, v3
# Inputs to the model
x1 = torch.randn(1, 4, 8, 8)
