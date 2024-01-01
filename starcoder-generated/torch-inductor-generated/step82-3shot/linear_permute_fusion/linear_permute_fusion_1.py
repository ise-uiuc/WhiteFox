
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 3)
        self.batch_norm = torch.nn.BatchNorm2d(1)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = self.batch_norm(v1)
        v3 = v2.permute(0, 2, 1)
        v4 = v3.permute(0, 2, 1)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 1)
