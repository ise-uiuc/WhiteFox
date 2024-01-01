
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.conv = torch.nn.Conv2d(1, 1, (1, 1), 1, (0,), 1, 1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v2 = torch.tanh(v2)
        v3 = v2.unsqueeze(1)
        v4 = self.conv(v3)
        v5 = v4.squeeze(1)
        v6 = v2 + v5
        return v6 + v2
# Inputs to the model
x1 = torch.randn(1, 3, 3)
