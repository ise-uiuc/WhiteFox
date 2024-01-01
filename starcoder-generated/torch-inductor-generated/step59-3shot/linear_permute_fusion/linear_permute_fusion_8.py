
class Model(torch.nn.Module):
    def __init__(self, inp):
        super().__init__()
        self.linear = torch.nn.Linear(inp, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = x1.permute(0, 2, 1)
        return self.linear(v2)
# Inputs to the model
x1 = torch.randn(1, 3, 2)
