
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, return_indices=False, ceil_mode=False)
    def forward(self, x1):
        v1 = self.pool1(x1)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 17, 10)
