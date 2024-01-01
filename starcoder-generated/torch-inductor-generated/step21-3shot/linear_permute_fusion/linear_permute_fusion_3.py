
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v3 = x1[:, :, 0]
        v2 = x1[:, :, 1]
        v1 = torch.cat([torch.cat([v3, v3], 1), torch.cat([v2, v2], 1)], 0)
        v1 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        return v2
# Inputs to the model
x1 = torch.randn(2, 2, 3)
