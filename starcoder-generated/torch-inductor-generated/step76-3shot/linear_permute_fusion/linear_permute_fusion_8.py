
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 3)
    def forward(self, x0):
        v0 = self.linear1(x0)
        v1 = torch.nn.functional.linear(v0, self.linear1.weight, self.linear1.bias)
        v2 = v1.permute(1, 0)
        return v2
# Inputs to the model
x0 = torch.randn(1, 2)
