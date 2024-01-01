
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = torch.nn.MaxPool2d(3, stride = 1, padding=1)
    def forward(self, x):
        v1 = self.max_pool(x)
        return v1
# Inputs to the model
x = torch.randn(1, 3, 256, 256)
