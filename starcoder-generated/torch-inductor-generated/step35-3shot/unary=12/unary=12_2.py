
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(30, 16, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=1)

    def forward(self, input_tensor):
        v1 = self.conv1(input_tensor)
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
input_tensor = torch.randn(1, 30, 64, 64)
