
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.ConvTranspose2d(3, 32, 5, stride=1, padding=1)
        self.conv_2 = torch.nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1)
    def forward(self, x1):
        v6 = self.conv_1(x1)
        v7 = self.conv_2(v6)
        v8 = torch.relu(v7)
        v9 = torch.sigmoid(v8)
        v10 = torch.tanh(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
