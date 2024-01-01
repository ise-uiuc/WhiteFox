
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
          torch.nn.Conv2d(3, 4, 5, stride=1, padding=2),
          torch.nn.ReLU(True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.BatchNorm2d(4),
          torch.nn.Conv2d(4, 8, 10, stride=1, padding=4),
          torch.nn.ReLU(True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.BatchNorm2d(8),
          torch.nn.Flatten(),
          torch.nn.Linear(1024, 20),
          torch.nn.ReLU(True),
          torch.nn.BatchNorm1d(20),
          torch.nn.Linear(20, 3),
          torch.nn.Softmax())
    def forward(self, x1):
        v1 = self.layers(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
