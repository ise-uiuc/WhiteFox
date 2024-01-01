
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 1, 1, stride=1, padding=0)
        self.sigmoid2 = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = torch.relu(v1)
        v3 = self.sigmoid2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
