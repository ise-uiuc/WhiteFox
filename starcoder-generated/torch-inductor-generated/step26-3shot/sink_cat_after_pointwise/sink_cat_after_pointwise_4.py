
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = torch.nn.Linear(1, 2)
        self.m2 = torch.nn.Linear(10, 10)
    def forward(self, x):
        x = torch.cat([x, x], dim=1)
        x = torch.relu(x + self.m1(x))
        return self.m2(x)
# Inputs to the model
x = torch.randn(3, 2, 2)
