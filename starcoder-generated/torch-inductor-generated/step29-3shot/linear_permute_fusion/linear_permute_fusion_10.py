
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, 3)
        self.linear2 = torch.nn.Linear(3, 3, bias=False)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1 + 2.0, self.linear1.weight)
        v2 = torch.nn.functional.linear(x1 + 0.5, self.linear2.weight)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 1)
