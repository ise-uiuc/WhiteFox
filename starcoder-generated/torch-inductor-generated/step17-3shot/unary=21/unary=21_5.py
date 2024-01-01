
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(4, 6, 2)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)
        self.conv3 = torch.nn.Conv2d(1, 1, 1)
        self.conv4 = torch.nn.Conv2d(3, 5, 7, stride=(2, 1), padding=(2, 3), dilation=(1, 1))
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = self.conv3(v1)
        v4 = self.conv4(v2)
        v5 = torch.tanh(v4)
        return v5
# Inputs to the model
x = torch.randn(35, 4, 26)
