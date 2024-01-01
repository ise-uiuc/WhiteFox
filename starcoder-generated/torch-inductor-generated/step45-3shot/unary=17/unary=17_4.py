
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(128, 64, 1, padding=0, stride=4)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(64, 128, 3, padding=1, stride=2)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(128, 64, 1, padding=0, stride=2)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(64, 1, 5, padding=0, stride=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.relu(v1)
        v3 = self.conv_transpose1(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv_transpose2(v4)
        v6 = torch.relu(v5)
        v7 = self.conv_transpose3(v6)
        v8 = torch.relu(v7)
        v9 = torch.sigmoid(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 128, 64, 64)
