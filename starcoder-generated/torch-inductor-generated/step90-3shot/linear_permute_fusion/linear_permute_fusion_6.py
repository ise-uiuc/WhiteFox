
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x3):
        v3 = torch.nn.functional.linear(x3, self.linear.weight, self.linear.bias)
        v4 = v3.permute(0, 2, 1)
        return v4
# Inputs to the model
x3 = torch.randn(1, 3, 2)
