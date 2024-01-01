
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convt2d = torch.nn.ConvTranspose2d(3, 2, 5, padding=8, output_padding=7)
    def forward(self, x1):
        v1 = self.convt2d(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 5, 13)
