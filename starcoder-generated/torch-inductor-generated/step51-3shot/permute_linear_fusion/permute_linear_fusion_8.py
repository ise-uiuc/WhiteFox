
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.permute = torch.nn.Permute([0, 2, 1])
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = self.permute(x1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        return v2.permute(0, 2, 1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
