
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(5, 1, 19, stride=1, padding=0)
        self.conv2 = torch.nn.Conv1d(1, 1, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv1d(1, 6, 5, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        return v3
# Inputs to the model
x1 = torch.randn(2, 5, 54)
