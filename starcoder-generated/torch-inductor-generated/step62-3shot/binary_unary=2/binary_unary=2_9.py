
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 32, kernel_size=(1, 3))
    def forward(self, x0):
        v0 = self.conv2d(x0)
        v1 = v0 - 1
        v2 = F.relu(v1)
        return v2
# Inputs to the model
x0 = torch.randn(1, 3, 32, 32)
