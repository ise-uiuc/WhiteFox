
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self, x0):
        x1 = self.conv2d(x0)
        v1 = torch.nn.functional.relu(x1)
        return v1
kernel_size = 3
# Inputs to the model
x0 = torch.randn(1, 3, 32, 32)
