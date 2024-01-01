
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(1, 16, (6, 3), padding=(0, 0), stride=(1, 2))
        self.conv2 = torch.nn.ConvTranspose2d(16, 32, (3, 7), padding=(0, 0), stride=(1, 3))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
