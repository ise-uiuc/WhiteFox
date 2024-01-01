
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(4, 48, 7, stride=2, padding=3, bias=False)
        self.conv2 = torch.nn.Conv1d(48, 48, 3, stride=1, padding=1, bias=True)
        self.conv3 = torch.nn.Conv1d(48, 48, 3, stride=2, padding=1, bias=False)
        self.conv4 = torch.nn.Conv1d(48, 96, 3, stride=1, padding=2, bias=True)
        self.conv5 = torch.nn.Conv1d(96, 96, 1, stride=2, padding=0, bias=True)
    def forward(self, x1):
        v1 = torch.relu(self.conv1(x1))
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 4, 128)
