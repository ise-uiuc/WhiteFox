
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(5, 8, 3, stride=1, padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(8, 32, 3, stride=1, padding=1)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(32, 16, 3, stride=1, padding=1)
        self.conv_transpose4 = torch.nn.ConvTranspose2d(16, 32, 3, stride=1, padding=1)
        self.conv_transpose5 = torch.nn.ConvTranspose2d(32, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = F.relu(v1)
        v3 = self.conv_transpose2(v2)
        v4 = F.relu(v3)
        v5 = self.conv_transpose3(v4)
        v6 = F.relu(v5)
        v7 = self.conv_transpose4(v6)
        v8 = F.relu(v7)
        v9 = self.conv_transpose5(v8)
        v10 = F.relu(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
