
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = torch.nn.ConvTranspose2d(3, 4, 2, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.deconv1(x1)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 248, 248)
