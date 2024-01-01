
input_tensor = None
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 4, 3, stride=2, padding=1)
    def forward(self, x):
        x = x * 1.2
        output = self.conv(x)
        return output
# Inputs to the model
x = torch.randn(1, 2, 35, 35)
input_tensor = x
