
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
        self.linear1 = torch.nn.Linear(2, 2, bias=False)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1.permute(0, 2, 1), self.linear.weight, self.linear.bias)
        return self.linear1(v1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
