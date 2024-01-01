
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
    def forward(self, x3):
        v3 = torch.nn.functional.linear(x3, self.linear.weight, self.linear.bias)
        v4 = v3.permute(0, 2, 1)
        return v3
# Inputs to the model
x3 = torch.randn(3, 2, 2)
