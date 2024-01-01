
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose1d(4, 4, 3, stride=2, padding=1)
        self.conv1 = torch.nn.ConvTranspose1d(4, 4, 3, stride=2, padding=1)

    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv1(v2)
        v4 = torch.nn.functional.relu(v3)
        v5 = torch.nn.functional.relu6(v2)
        v6 = torch.tanh(v4)
        v7 = torch.sigmoid(v6)
        v8 = torch.nn.functional.hardtanh(v7)
        v9 = torch.tanh(v8)
        return v6
# Input to the model
x1 = torch.randn(1, 4, 20, dtype=torch.float32)
