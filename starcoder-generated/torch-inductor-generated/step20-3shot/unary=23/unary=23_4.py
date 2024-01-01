
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose2 = torch.nn.ConvTranspose2d(200, 200, kernel_size=[147, 147])
    def forward(self, x1):
        v1 = self.conv_transpose2(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 200, 213, 213)
