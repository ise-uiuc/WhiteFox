
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 1, stride=1, padding=1)
    def forward(self, x1, input_tensor):
        v1 = self.conv(x1)
        v2 = input_tensor - 1e-06
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
input_tensor = torch.randn(1, 3, 64, 64)
