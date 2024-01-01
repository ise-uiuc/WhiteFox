
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 64, (2, 2), stride=(2, 2))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.tensor([123.0, 456.0, 789.0])
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(2, 3, 3, 3)
