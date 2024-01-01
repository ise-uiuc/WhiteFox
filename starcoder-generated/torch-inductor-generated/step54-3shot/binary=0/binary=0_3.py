
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 2, stride=2, groups=1, dilation=1, padding=0)
    def forward(self, input_x):
        output_y = self.conv(input_x)
        return output_y
# Inputs to the model
input_x = torch.randn(1, 3, 64, 64)
