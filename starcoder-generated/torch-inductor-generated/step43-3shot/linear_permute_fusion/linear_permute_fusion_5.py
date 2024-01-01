
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = torch.nn.Linear(3, 1)
        self.linear1 = torch.nn.Linear(1, 2)
    def forward(self, x0):
        v0 = torch.nn.functional.linear(x0, self.linear0.weight, self.linear0.bias)
        v1 = v0.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        return v2
# Inputs to the model
x0 = torch.randn(1, 3, 3) * 4
