
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.bn2d = torch.nn.BatchNorm1d(1)
    def forward(self, x):
        v1 = x.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v2.unsqueeze(1)
        v4 = self.bn2d(v3)
        v5 = v4.squeeze(1)
        v6 = v2 + v5
        return v6
# Inputs to the model
x = torch.randn(1, 2, 2)
