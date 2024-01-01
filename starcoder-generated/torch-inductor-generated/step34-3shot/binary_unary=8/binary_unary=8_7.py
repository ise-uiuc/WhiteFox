
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, (3, 5), stride=(2, 3), padding=(1, 2))
        self.conv2 = torch.nn.Conv1d(3, 4, 5, stride=3, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = torch.relu(v1 + v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
