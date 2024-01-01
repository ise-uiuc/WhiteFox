
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(32, 32)
        self.linear2 = torch.nn.Linear(32, 1)
    def forward(self, x1):
        v1 = x1.permute(1, 0)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        return torch.nn.functional.linear(v1, self.linear2.weight, self.linear2.bias)
# Inputs to the model
x1 = torch.randn(32, 1)
