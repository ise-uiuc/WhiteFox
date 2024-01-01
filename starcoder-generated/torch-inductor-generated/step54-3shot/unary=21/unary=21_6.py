
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Conv-ReLU-MaxPool layer
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 128, 3, stride=1, padding=1))
    def forward(self, x):
        return torch.tanh(self.features(x))
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
