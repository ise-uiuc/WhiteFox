
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(stride=3, kernel_size=7, padding=2)
    def forward(self, x1):
        v1 = self.pool(x1)
        v2 = torch.squeeze(v1, 0)
        v3 = v2 - 0.5
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
