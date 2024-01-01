
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(2, 2, 2, groups=2, bias=False)
    def forward(self, x1):
        y = torch.max(x1, dim=-1, keepdim=True)[0]
        y = y.permute(0, 2, 1)
        v1 = self.conv(x1)
        z = v1 / y
        y = y.permute(2, 0, 1)
        return torch.nn.functional.relu(z - y)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
