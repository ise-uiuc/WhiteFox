
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(32, 3, 3, stride=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.reshape(torch.reshape(v1, (-1, 64, 100, 150)), (8000, 3))  # Aux input for the output of another layer
        return v2
# Inputs to the model
x1 = torch.randn(1, 32, 20, 10)
