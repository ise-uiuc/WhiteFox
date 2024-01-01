
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool_t = torch.nn.MaxPool2d(4,stride=2, padding=2)
    def forward(self, x1):
        v1 = self.maxpool_t(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 64, 128, 128)
