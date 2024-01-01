
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.mean(v1 ** 2, dim=1)
        v3 = v2.unsqueeze(1) * 2
        v4 = v1 + v3
        v5 = torch.nn.functional.linear(v4, self.linear.weight * 2, self.linear.bias)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
