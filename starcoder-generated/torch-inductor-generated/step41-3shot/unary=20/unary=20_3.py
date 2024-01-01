
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu_t = torch.nn.ConvTranspose2d(256, 16, kernel_size=1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.relu_t(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 256, 112, 112)
