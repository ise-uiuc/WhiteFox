
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(16, 24, 3, stride=2, padding=1)
        self.batch_norm = torch.nn.BatchNorm1d(24)
    def forward(self, x1):
        v1 = self.batch_norm(self.conv(x1))
        v2 = v1 - 4.2
        v3 = F.relu(v2)
        v4 = torch.chunk(v3, 2, dim=1)
        return v4[0]
# Inputs to the model
x1 = torch.randn(1, 16, 32)
