
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.ReLU()
        self.b = torch.nn.ReLU()
    def forward(self, x):
        y = torch.relu(x)
        y = torch.cat((y, y), dim=1)
        x = y.tanh()
        x = self.a(x)
        x = y.tanh()
        x = self.b(x)
        x = x * y
        return x
# Inputs to the model
x = torch.randn(2, 2, 2)
