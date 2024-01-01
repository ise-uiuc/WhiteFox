
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
    def forward(self, x1):
        v1 = x1
        v2 = v1.permute(*torch.arange(v1.ndim)[[1, -2, 0]])
        v3 = torch.nn.functional.linear(v2, self.linear.weight, self.linear.bias)
        v4 = v3.permute(v3.shape[1], -2, *range(v3.ndim)[2:0:-1])
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 3)
