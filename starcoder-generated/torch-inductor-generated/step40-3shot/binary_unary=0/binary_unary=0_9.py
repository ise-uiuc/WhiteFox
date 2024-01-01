
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 32, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(32, 64, 7, stride=1, padding=3)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = torch.cat([v2, x2], dim=1)
        v4 = self.conv2(v3)
        v5 = v4 * x2
        v6 = torch.nn.functional.avg_pool2d(v5, 2, stride=1)
        _, max_idx = torch.max(v6.view(1, -1), 1)
        max_y = (max_idx // 8) * 2
        max_x = (max_idx % 8) * 2
        return v5[:, :, max_y:(max_y + 7), max_x:(max_x + 7)]
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
