
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 3, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = torch.sigmoid(v2)
        v4 = torch.tanh(v3)
        v5 = v4.squeeze(dim=0)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
