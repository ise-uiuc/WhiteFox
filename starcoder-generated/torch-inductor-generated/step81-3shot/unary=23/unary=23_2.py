
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(49, 128, 11, stride=1, padding=0, output_padding=0)
        # self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        # v2 = self.relu(v1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 49, 10, 10)
