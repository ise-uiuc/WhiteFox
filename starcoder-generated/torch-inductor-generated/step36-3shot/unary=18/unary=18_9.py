
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=64, out_channels=7, kernel_size=1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv1d(in_channels=7, out_channels=3, kernel_size=1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 64, 80)
