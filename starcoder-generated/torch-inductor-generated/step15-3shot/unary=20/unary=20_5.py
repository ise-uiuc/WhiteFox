
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.t = ops.ConvTranspose(3, 8, kernel_size=(3, 3))
    def forward(self, x1):
        v1 = self.t(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(10, 3, 12, 12)
