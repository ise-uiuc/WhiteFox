
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(167223, 150)
    def forward(self, x1):
        v1 = self.flatten(x1)
        v2 = self.linear(v1)
        return v2
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
