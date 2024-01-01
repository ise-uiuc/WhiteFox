
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 3, (1, 2), stride=(1, 2), padding=(1, 2))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = F.relu(v1)
        v3 = F.relu(v2)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
