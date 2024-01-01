
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(144, 8, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1))
    def forward(self, input, padding=None):
        v1 = self.conv2d(input)
        v2 = v1 + padding
        return v2
# Inputs to the model
input = torch.randn(1, 144, 8, 8)
