
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(5, 5, 2).to(memory_format=torch.channels_last)
        self.bn = torch.nn.BatchNorm1d(5).to(memory_format=torch.channels_last)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))
# Inputs to the model
x = torch.randn(10, 5, 64)
