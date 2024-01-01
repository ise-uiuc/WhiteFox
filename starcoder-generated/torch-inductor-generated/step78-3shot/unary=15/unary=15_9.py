
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 3, (1, 1)), torch.nn.ReLU())
    def forward(self, x1):
        v0 = x1
        v1 = self.features(v0)
        v3 = torch.max_pool2d(v1, kernel_size=(28, 28))
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
