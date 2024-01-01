
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 8, (3, 3), stride=(2, 2))
        self.conv_transpose1 = torch.nn.ConvTranspose2d(8, 16, (3, 3), stride=(2, 2))
        self.conv_transpose2 = torch.nn.ConvTranspose2d(16, 32, (3, 3), stride=(2, 2))
        self.conv_transpose3 = torch.nn.ConvTranspose2d(32, 64, (3, 3), stride=(2, 2))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.relu(v1)
        v3 = self.conv_transpose1(v2)
        v4 = torch.relu(v3)
        v5 = self.conv_transpose2(v4)
        v6 = torch.relu(v5)
        v7 = self.conv_transpose3(v6)
        v8 = torch.relu(v7)
        v9 = torch.sigmoid(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 1, 4, 4)
