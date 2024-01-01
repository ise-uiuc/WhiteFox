
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 2, (3, 4), stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
    def forward(self, x1):
        v1 = torch.sigmoid(self.conv1(x1))
        v2 = self.pool(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
