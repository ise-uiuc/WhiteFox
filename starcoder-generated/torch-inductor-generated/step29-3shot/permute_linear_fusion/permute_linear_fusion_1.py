
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v2 * v1
        v4 = v3.mean(dim=0)
        v4 = v4.sum(dim=-1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
