
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 2), stride=(2, 2), padding=(2, 1), dilation=(3, 2))
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        y1, _ = torch.max(v2, dim=-1)
        v3 = y1.permute(0, 2, 1).expand(-1, 3, -1)
        v4 = y1.unsqueeze(dim=-1).expand(-1, -1, 3)
        z3 = torch.stack([v3, v4], dim=-1)
        v5 = self.conv(z3)
        x2 = v2 + v5
        return x2
# Inputs to the model
x1 = torch.randn(1, 3, 3)
