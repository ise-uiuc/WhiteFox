
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(192, 3, 3, padding=1, stride=1)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 2, 3, padding=1, stride=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.relu(v1)
        v3 = self.conv_transpose1(v2)
        v4 = torch.relu(v3)
        v5 = torch.sigmoid(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 192, 224, 224)
