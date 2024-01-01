
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(12, 96, 21, stride=7, padding=1, output_padding=2)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.nn.functional.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 12, 224, 224)
