
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x2):
        v2 = torch.nn.functional.linear(x2, self.linear.weight, self.linear.bias)
        v3 = v2.permute(0, 2, 1)
        return v3
# Inputs to the model
x2 = torch.randn(1, 3, 2)
