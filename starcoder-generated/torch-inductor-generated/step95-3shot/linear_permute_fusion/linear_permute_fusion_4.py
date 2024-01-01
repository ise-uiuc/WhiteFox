
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 3)
        self.linear2 = torch.nn.Linear(3, 5)
    def forward(self, x0):
        v0 = torch.nn.functional.linear(x0, self.linear1.weight, self.linear1.bias)
        linear2 = torch.nn.Linear(3, 5)
        v1 = linear2(v0)
        return v1
# Inputs to the model
x0 = torch.randn(1, 15, 3)
