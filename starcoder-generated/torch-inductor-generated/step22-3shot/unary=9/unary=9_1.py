
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 9)
    def forward(self, x1):
        x1 = torch.nn.functional.hardtanh(x1 + 3.0, min_val = -3.0, max_val = 0.0)
        x1 = torch.nn.functional.interpolate(x1, scale_factor=0.0625, mode='nearest')
        x1 = self.conv(x1)
        x1 = torch.nn.functional.relu6(x1)
        x1 = torch.nn.functional.softmax(x1, dim=0)
        return x1
# Inputs to the model
x1 = torch.randn(2, 3, 128, 128)
