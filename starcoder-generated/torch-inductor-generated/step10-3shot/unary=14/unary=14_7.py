
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtranspose = torch.nn.ConvTranspose2d(3, 3, kernel_size=(1, 1))
    def forward(self, x1):
        v1 = self.convtranspose(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3, v2, v1
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32) # (N, C_in, H_in, W_in)
