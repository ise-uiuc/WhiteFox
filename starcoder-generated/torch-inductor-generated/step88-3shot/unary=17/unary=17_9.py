
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(512, 512, (1, 1), stride=(1, 1))
        self.conv_transpose = torch.nn.ConvTranspose2d(512, 512, (1, 1), stride=(1, 1))
        self.conv_2 = torch.nn.Conv2d(512, 10, 1)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv_transpose(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv_2(v4)
        v6 = torch.sigmoid(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 512, 7, 7)
