
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
    def forward(self, x1):
        x2 = torch.matmul(x1.permute(0, 2, 1),
                          self.linear.weight) * x1
        y = torch.matmul(x1.permute(0, 2, 1), self.linear.bias)
        return torch.sigmoid(x2 + x1 * y)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
