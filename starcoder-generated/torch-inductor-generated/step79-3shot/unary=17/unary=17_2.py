
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 6, 3, padding=2, stride=1)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(6, 12, 3, padding=2, stride=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(12, 18, 3, padding=2, stride=1)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(18, 32, 3, padding=2, stride=1)
        self.conv_transpose4 = torch.nn.ConvTranspose2d(32, 24, 3, padding=2, stride=1)
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
        v11 = torch.sigmoid(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 1, 1, 1)
