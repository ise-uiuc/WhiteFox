
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtranspose_1_2 = torch.nn.ConvTranspose1d(256, 512, 496, stride=2, padding=128, output_padding=247)
    def forward(self, x1):
        v1 = self.convtranspose_1_2(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 256, 496)
