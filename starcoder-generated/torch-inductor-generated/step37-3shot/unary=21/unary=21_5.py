
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(128, 2, kernel_size=1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(1, 8, 2, stride=(0, 1), padding=0)
        self.conv3 = torch.nn.Conv2d(8, 3, 2, stride=(0, 1), padding=1)
    def forward(self, x):
        v1 = self.conv1(x).reshape(1, -1)
        v2 = torch.tanh(v1)
        return self.conv3(self.conv2(v2).reshape(8*4, 120, 1, 2)).reshape(8, 4, 120, 2)
# Inputs to the model
x = torch.randn(1, 2, 128, 25)
