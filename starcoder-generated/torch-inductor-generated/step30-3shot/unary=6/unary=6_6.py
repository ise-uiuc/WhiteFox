
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(100, 200, bias=True)
        self.conv = torch.nn.Conv2d(3, 3, 5, stride=5, padding=1, bias=False)
    def forward(self, x):
        x = torch.relu(self.linear(x))
        return torch.relu(self.conv(x))

# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
x2 = torch.randn(1, 100)
