
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(input_channels = 3, output_channels = 21, kernel_size = (2, 2), stride = (2, 2),  bias = False)
    def forward(self, x1):
        v1 = torch.nn.functional.relu(x1 = self.conv(x1 = x1))
        return v1
# Inputs to the model
x1 = torch.randn(38, 3, 32, 32)
