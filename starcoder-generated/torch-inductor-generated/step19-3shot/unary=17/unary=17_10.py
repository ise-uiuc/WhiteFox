
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 16, 3, padding=1, stride=2)
        self.linear = torch.nn.Linear(205440/4, 8)
    def forward(self, x1):
        v1 = self.conv(x1)
        t1 = v1.reshape((v1.shape[0], -1))
        v2 = self.linear(t1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
