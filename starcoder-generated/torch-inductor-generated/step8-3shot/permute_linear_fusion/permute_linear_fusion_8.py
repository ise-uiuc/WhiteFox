
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.flatten = torch.nn.Flatten(0, 1)
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        v3 = self.flatten(v2)
        v4 = torch.nn.functional.linear(v3, self.linear2.weight, self.linear2.bias)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
