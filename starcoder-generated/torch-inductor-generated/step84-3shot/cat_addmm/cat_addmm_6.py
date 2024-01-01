
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(2, 3)
        self.b = torch.nn.Linear(2, 3)
    def forward(self, x):
        x = self.a(x)
        x = torch.add(x, x)
        x = self.b(x)
        return x
# Inputs to the model
x = torch.randn(2, 2)
