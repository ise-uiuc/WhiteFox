
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(3, 31, 1, stride=1, padding=0, dilation = 2)
        self.conv2 = torch.nn.Conv1d(31, 72, 3, stride=2, padding=1, dilation = 3)
        self.conv3 = torch.nn.Conv1d(72, 144, 5, stride=2, padding=2, dilation = 4)
        self.conv4 = torch.nn.Conv1d(144, 288, 7, stride=3, padding=3, dilation = 5)
        self.conv5 = torch.nn.Conv1d(288, 576, 8, stride=2, padding=1, dilation = 8)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 3200)
