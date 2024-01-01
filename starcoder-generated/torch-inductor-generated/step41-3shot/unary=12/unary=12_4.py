
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(1, 4, (1, 1), stride=1, padding=(0, 0), dilation=1)
        self.relu = torch.nn.ReLU6()
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = self.relu(v1)
        return v2
# Inputs to the model
input_data = [torch.randn(1, 1, 4, 32)]
