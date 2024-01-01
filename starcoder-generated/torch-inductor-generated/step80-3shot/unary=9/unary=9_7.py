
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, (1, 1), stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.conv_2 = torch.nn.Conv2d(3, 8, (1, 1), stride=1, padding=1)
        self.hardtanh = torch.nn.Hardtanh()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.relu(v1)
        v3 = self.conv_2(v2)
        v4 = self.hardtanh(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
