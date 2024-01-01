
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
    def forward(self, x1):
        v1 = self.max_pool(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=6)
        v5 = v1 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 15, 15)
