
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 1, 2)
        v2 = torch.nn.functional.linear(v1, self.linear.weight.T, self.linear.bias)
        return self.linear.weight + v1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
