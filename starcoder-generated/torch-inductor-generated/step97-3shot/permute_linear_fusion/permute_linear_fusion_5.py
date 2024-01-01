
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 5)
        self.linear2 = torch.nn.Linear(5, 2)
    def forward(self, x1):
        v0 = x1.permute(0, 2, 1)
        v1 = torch.nn.functional.linear(v0, self.linear1.weight, self.linear1.bias)
        v2 = torch.nn.functional.relu(v1)
        v3 = torch.nn.functional.linear(v2, self.linear2.weight, self.linear2.bias)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
