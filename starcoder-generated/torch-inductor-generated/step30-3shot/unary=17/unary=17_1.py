
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.MaxPool2d(2, 2, return_indices=True, ceil_mode=True)
    def forward(self, x1):
        v1, v2 = self.conv(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
