
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(1, 46, 1, bias=False))
    def forward(self, x):
        v1 = torch.tanh(self.features(x))
        return v1
# Inputs to the model
x = torch.randn(55, 1, 90, 20)
