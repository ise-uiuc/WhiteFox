
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.Conv2d(3, 24, 3, padding=1, stride=2)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(24, 24, 3, padding=1, stride=2)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(24, 24, 3, padding=1, stride=1)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(24, 24, 3, padding=1, stride=2)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.relu(v1)
        v3 = torch.sigmoid(v2)

        v4_1 = self.conv_transpose1(v3)
        v4_2 = v4_1

        v5_1 = self.conv_transpose2(v4)
        v5_2 = torch.sigmoid(v5_1)

        v6_1 = self.conv_transpose3(v4_2)
        v6_2 = v6_1
        v6_3 = torch.sigmoid(v6_2)
        return v6_3
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
