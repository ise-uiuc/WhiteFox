
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1) / (self.linear.weight + self.linear.bias)
        y = (v1 * 2) * x1

        x2 = self.linear.weight + x1
        y = self.linear.bias * (x2 * 3)
        return y / x1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
