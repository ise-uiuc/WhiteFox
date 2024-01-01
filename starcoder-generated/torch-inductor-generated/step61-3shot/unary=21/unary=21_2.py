
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(1, 22, 1, stride=1, padding=0))
    def forward(self, x):
        return torch.tanh(self.features(x))
# Inputs to the model
x = torch.randn(120, 1, 80, 20)
