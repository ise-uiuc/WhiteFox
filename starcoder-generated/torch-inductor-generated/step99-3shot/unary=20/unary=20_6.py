
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(195, 64, (75, 24), (11, 30), (29, 20), 2)
    def forward(self, x1):
        v1 = torch.sigmoid(self.conv_t(x1))
        return v1
# Inputs to the model
x1 = torch.randn(1, 195, 17, 48)
