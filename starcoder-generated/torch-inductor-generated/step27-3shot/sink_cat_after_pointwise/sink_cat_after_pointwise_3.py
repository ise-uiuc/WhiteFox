
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2000, 1000)
    def forward(self, x):
        y = x * 4
        y = y.repeat(1, 2000)
        u = torch.relu(y)
        v = torch.cat((u, u), dim=1)
        w = v.view(8, 2, -1)
        x = w.transpose(1, 2)
        return self.linear(x)
# Inputs to the model
x = torch.randn(1, 2000, 1)
