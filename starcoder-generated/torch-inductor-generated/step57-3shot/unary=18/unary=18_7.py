
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = torch.nn.MaxPool2d(3, 1, 1)
    def forward(self, x1):
        v1 = self.max_pool(x1)
        v2 = torch.sigmoid(v1)
        return v2
x1 = torch.randn(1, 512, 7, 7)
