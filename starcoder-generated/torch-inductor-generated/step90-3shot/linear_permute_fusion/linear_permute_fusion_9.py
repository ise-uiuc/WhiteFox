
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        return torch.ops.aten.avg_pool2d_backward(v1, x1, (1, 1), (1, 1))
# Inputs to the model
x1 = torch.randn(1, 2, 3)
