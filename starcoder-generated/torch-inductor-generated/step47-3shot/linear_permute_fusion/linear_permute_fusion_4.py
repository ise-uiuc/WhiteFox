
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, 3)
        self.linear2 = torch.nn.Linear(4, 10, bias=False)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear1.weight, self.linear1.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = torch.nn.functional.linear(v2, self.linear2.weight, None)
        return v3

# Inputs to the model
x1 = torch.randn(2, 10, 1)
