
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.nn.functional.max_pool2d(x1, kernel_size = 1, stride = 1, padding = 1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
