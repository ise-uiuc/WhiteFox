
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.TConv = torch.nn.ConvTranspose2d(1, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.flatten(v1, 1)
        v3 = v2.view(-1, 1, 128, 128)
        v4 = self.TConv(v3)
        v5 = F.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
