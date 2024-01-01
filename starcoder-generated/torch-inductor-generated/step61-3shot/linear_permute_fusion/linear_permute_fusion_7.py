
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 3)
        self.linear2 = torch.nn.Linear(3, 2)
    def forward(self, x1):
        v2 = torch.nn.functional.linear(x1, self.linear1.weight, self.linear1.bias)
        v3 = torch.nn.functional.linear(v2, self.linear2.weight, self.linear2.bias)
        return v3.permute(0, 2, 1)
# Inputs to the model
x1 = torch.randn(1, 3, 3)
