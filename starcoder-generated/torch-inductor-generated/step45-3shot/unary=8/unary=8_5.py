
class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 6, 2, stride=0, padding=0, output_padding=0)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(3, 6, 2, stride=2, padding=1, output_padding=0)
        self.relu = torch.nn.ReLU()
        self.concat = torch.nn.functional.quantized.ConcatStub()

    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(x1)
        v3 = self.relu(v1)
        v4 = torch.cat([v3, v2], 1)
        v5 = v1 + 3
        v6 = torch.clamp(v5, min=0)
        v7 = torch.clamp(v6, max=6)
        v8 = v3 * v7
        v9 = torch.floor_divide(v8, 4)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
