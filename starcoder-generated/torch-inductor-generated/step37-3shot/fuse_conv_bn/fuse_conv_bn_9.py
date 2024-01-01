
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(32, 32, 3, padding=(2, 1, 0), stride=(2, 2, 2), bias=False)
    def forward(self, x1):
        x2 = F.relu(x1)
        x3 = F.avg_pool3d(x2, (3, 3, 3), stride=1, padding=1, count_include_pad=True)
        x3 = F.relu(x3)
        x4 = self.conv(x3)
        x5 = F.relu(x4)
        x6 = F.avg_pool3d(x5, (1, 3, 3), stride=1, padding=(1, 1, 1), count_include_pad=False)
        x7 = F.sigmoid(x6)
        return x7
# Inputs to the model
x1 = torch.randn(1, 32, 22, 22, 12)
