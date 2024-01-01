
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.conv1 = torch.nn.Conv2d(1, 1, 2, bias=False)
        self.conv2 = torch.nn.Conv2d(1, 1, 2, bias=False)
    def forward(self, x, y):
        x = self.relu(self.conv1(x))
        y = self.tanh(self.conv2(y))
        z = torch.cat([x, y], dim=1)
        return z
# Inputs to the model
x = torch.randn(1, 1, 4, 4)
y = torch.randn(1, 1, 4, 4)
