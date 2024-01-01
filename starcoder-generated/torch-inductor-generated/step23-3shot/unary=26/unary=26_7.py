
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(8, 9, 2, stride=2)
    def forward(self, x1):
        w1 = self.conv_t(x1)
        w2 = torch.abs(w1)
        w3 = w1.flatten()
        w4 = (torch.sigmoid(w3) - 0.75) * 3
        return w2 + w4.reshape(8, 9, 2, 2)
# Inputs to the model
x1 = torch.randn(8, 8, 2, 2)
