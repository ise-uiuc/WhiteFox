
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 16, kernel_size=(5, 5), stride=(1, 1), padding=(5, 5))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 128.0
        v3 = F.relu(v2)
        v4 = torch.zeros_like(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 64, 20, 20)
