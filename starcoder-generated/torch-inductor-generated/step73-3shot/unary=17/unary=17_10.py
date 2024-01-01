
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(16, 64, 5, stride=2, padding=2, output_padding=0)
        self.conv1 = torch.nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=0)
        self.convt =torch.nn.ConvTranspose2d(64, 16, 2, stride=1, padding=0, output_padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = F.relu(v1)
        v3 = self.conv1(v2)
        v4 = F.relu(v3)
        v5 = self.convt(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 16, 14, 14)
