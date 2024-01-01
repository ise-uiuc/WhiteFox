
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1, 3, 4)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v2 = v2 * 2
        v3 = v2.reshape(1, 2, 4)
        v4 = torch.sum(v2, dim=[-1])
        v5 = v4.permute(0, 2, 1)
        return v2 + v5
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2, 2)
