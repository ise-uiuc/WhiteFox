
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=0)
        self.relu = torch.nn.ReLU6()
    def forward(self, input_tensor):
        x = self.conv1(input_tensor)
        v1 = self.relu(x)
        v2 = 6 - v1
        v3 = 3 + v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
