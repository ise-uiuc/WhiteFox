
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, 1)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear1.weight, self.linear1.bias)
        v2 = v1.permute(1, 0, 2)
        v3 = torch.nn.functional.linear(v2, self.linear1.weight, self.linear1.bias)
        return v3
# Inputs to the model
x1 = torch.randn(2, 1, 1)
