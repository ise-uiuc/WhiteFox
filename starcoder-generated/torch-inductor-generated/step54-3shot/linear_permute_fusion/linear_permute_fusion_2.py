
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 3)
        self.linear2 = torch.nn.Linear(2, 3)
    def forward(self, x2):
        v1 = torch.nn.functional.linear(x2, self.linear1.weight, self.linear1.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = torch.nn.functional.linear(v1, self.linear2.weight, self.linear2.bias)
        return v3
# Inputs to the model
x2 = torch.randn(1, 3, 2)
