
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x3):
        v3 = torch.nn.functional.linear(x3, self.linear1.weight, self.linear1.bias)
        v0 = v3.permute(1, 2, 0)
        v1 = torch.nn.functional.linear(v0, self.linear2.weight, self.linear2.bias)
        v2 = v1.permute(0, 2, 1).flatten(0, 1)
        return v2
# Inputs to the model
x3 = torch.randn(1, 2, 2)
