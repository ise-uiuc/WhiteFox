
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 6, 3, stride=2, padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(6, 16, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv_transpose2(v2)
        v4 = torch.relu(v3)
        v5 = v4.squeeze(dim=0)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
