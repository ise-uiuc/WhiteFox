
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.ReLU()
        self.b = torch.nn.ELU()
    def forward(self, x1):
        v1 = self.a(x1)
        v2 = self.b(x1)
        v3 = v1 + v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
