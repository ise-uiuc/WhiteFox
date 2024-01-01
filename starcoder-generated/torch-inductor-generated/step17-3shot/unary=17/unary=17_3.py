
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=torch.nn.ConvTranspose2d(3, 3, kernel_size=(3, 1), stride=(2, 1))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = LH.hswish(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
