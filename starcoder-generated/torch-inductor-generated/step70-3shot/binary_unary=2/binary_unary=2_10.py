
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.transpose = torch.nn.ConvTranspose2d(3, 3, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.transpose(v1)
        v3 = v2 - 0.5
        v4 = F.relu(v3)
        v5 = v4.unsqueeze(3)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
