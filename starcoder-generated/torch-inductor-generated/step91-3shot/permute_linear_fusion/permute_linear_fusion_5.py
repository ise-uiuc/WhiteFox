
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v2.abs()
        v4 = torch.all((v3 > 0).to(torch.int64), dim=-1)
        v5 = v4.permute(0, 2, 1)
        v5 = (v5 == 1).to(v5.dtype)
        x2 = v5.reshape(1, 2, 2)
        v6 = v5 * x2
        v7 = v5.permute(0, 2, 1)
        x3 = v6.sum(dim=1) > x3.max()
        return x3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
