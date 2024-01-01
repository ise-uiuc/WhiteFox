
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(1, 1, 4)
        self.conv2 = torch.nn.Conv1d(1, 1, 2, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v1)
        return v4
# Inputs to the model
x = torch.randn(1, 1, 128)
