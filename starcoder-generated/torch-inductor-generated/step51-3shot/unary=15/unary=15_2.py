
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2)
    def forward(self, x1):
        v1 = self.max_pool(x1)
        v2 = self.max_pool(v1)
        v3 = self.max_pool(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
