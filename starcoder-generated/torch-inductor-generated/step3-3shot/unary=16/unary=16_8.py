
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.relu(v1)
        return v2

x = torch.randn(1, 3)
y = x * x

x1 = torch.randn(1, 3, 50, 50)
