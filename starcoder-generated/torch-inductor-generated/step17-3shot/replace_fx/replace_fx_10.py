
class Model(torch.nn.Module):
    def __init__(self, a=1.0, b=2.0):
        super().__init__()
        self.a = a
        self.b = b
    def forward(self, x1):
        x2 = torch.mean(x1)
        x3 = torch.rand_like(x1)
        x4 = torch.zeros_like(x3).to(torch.double)
        x5 = torch.randn_like(x1)
        x6 = torch.randint_like(x1, high=10)
        x7 = self.a * x1 + self.b
        x10 = torch.where(x2 > 1, x2, torch.relu(x5))
        return x4 + x5 + x6 + x7 + x10
# Inputs to the model
x1 = torch.randint(low=1, high=100, size=(3, 4))
