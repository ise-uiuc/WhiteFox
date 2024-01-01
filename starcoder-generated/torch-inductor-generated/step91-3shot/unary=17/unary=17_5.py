
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(1, 32, 3, stride=2, padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(32, 48, 3, stride=2, padding=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v1_1 = v1.permute(0, 2, 3, 1)
        v1_2 = torch.softmax(v1_1, dim=-1)
        v1_3 = v1_2.permute(0, 3, 1, 2)
        v2 = self.conv2(v1_3)
        v3 = v2.view(-1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 56, 56)
