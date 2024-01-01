
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1_1 = torch.nn.ConvTranspose2d(16, 16, 7, stride=1, padding=0)
        self.conv_2_1 = torch.nn.ConvTranspose2d(16, 16, 7, stride=2, padding=3)
        self.conv_3 = torch.nn.ConvTranspose2d(16, 32, 7, stride=2, padding=3)
    def forward(self, x1):
        v1 = self.conv_1_1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv_2_1(v2)
        v4 = torch.tanh(v3)
        v5 = self.conv_3(v4)
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 16, 32, 32)
