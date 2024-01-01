
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv1d(64, 128, 64, stride=32)  # Pad = 0, Stride = 2, Kernel size = 7
        self.conv1 = torch.nn.Conv1d(128, 128, 50, stride=25)  # Pad = 0, Stride = 2, Kernel size = 5
    def forward(self, x):
        x = torch.relu(self.conv0(x))
        x = torch.tanh(self.conv1(x))
        return torch.max(x)
## Example Input
# input0 = torch.randn(1, 64, 100)
# input1 = torch.randn(1, 64, 100)
# input2 = torch.randn(1, 64, 100)
#