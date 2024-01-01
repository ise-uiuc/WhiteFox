
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 3, padding=2, stride=1)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(8, 12, 3, padding=2, stride=2)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(12, 16, 3, padding=1, stride=1)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(16, 32, 3, padding=1, stride=2)
        self.conv_transpose4 = torch.nn.ConvTranspose2d(32, 64, 3, padding=1, stride=2)
        self.conv_transpose5 = torch.nn.ConvTranspose2d(64, 128, 3, padding=1, stride=2)
        self.conv_transpose6 = torch.nn.ConvTranspose2d(128, 256, 3, padding=1, stride=2)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.relu(v1)
        v3 = self.conv_transpose1(v2)
        v4 = torch.relu(v3)
        v5 = self.conv_transpose2(v4)
        v6 = torch.relu(v5)
        v7 = self.conv_transpose3(v6)
        v8 = torch.relu(v7)
        v9 = self.conv_transpose4(v8)
        v10 = torch.relu(v9)
        v11 = self.conv_transpose5(v10)
        v12 = torch.relu(v11)
        v13 = self.conv_transpose6(v12)
        v14 = torch.relu(v13)
        return torch.sigmoid(v14)
# Inputs to the model
x1 = torch.randn(1, 3, 8, 8)
