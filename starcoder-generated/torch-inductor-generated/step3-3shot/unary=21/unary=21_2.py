
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d((3 + 4 * 5), 8, 1, stride=1, padding=1)
    def forward(self, input):
        x = self.conv(input)
        return x
# Inputs to the model
input = torch.randn(1, 3, 64, 64)
