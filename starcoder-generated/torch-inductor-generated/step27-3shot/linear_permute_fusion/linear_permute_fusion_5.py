
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        y = x1
        v1 = torch.nn.functional.linear(y, self.linear.weight, self.linear.bias)
        z = v1.permute(0, 2, 1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
