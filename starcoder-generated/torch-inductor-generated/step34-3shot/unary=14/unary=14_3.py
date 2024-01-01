
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 3, (7, 3), stride=(3, 5), padding=(1, 2))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.nn.Sigmoid()(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 274, 398)
