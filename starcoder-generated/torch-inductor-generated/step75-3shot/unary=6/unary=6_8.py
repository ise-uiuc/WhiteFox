
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.functional.conv2d
        self.relu = torch.nn.functional.relu
    def forward(self, x1):
        v1 = self.conv(input=x1, weight=torch.rand(192, 3, 1, 1), stride=1, padding=0, groups=1)
        v2 = v1 + 3
        v3 = self.relu(v2)
        v4 = torch.clamp_min(v3, 0)
        v5 = torch.clamp_max(v4, 6)
        return v5.unsqueeze(-1)
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
