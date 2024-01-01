
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x2, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = torch.nn.functional.linear(x2, self.linear.weight, self.linear.bias)
        v4 = v3.permute(2,0,1)
        return v4
# Inputs to the model
x2 = torch.randn(1, 2, 2)
x1 = torch.randn(1, 2, 2)
