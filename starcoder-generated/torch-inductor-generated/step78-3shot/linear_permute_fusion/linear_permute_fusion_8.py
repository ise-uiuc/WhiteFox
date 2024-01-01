
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
    def forward(self, x):
        v1 = torch.nn.functional.linear(x, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0)
        return v1 + v2
# Inputs to the model
x = torch.randn(1, 2, 2)
