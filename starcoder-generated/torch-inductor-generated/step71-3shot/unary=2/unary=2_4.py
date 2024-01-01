
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(2, 64, (1, 6), stride=(8, 4), padding=0)
        self.add2d = torch.nn.Add()
    def forward(self, x1, x2):
        v1 = self.conv_transpose(x1)
        v2 = x2 + 0.044715
        v3 = v2 * 0.5
        v4 = v1 + v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 10, 12)
x2 = torch.randn(1, 2, 10, 12)
