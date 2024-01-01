
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(5, 5, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        return nn.Sigmoid()(v1)
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
