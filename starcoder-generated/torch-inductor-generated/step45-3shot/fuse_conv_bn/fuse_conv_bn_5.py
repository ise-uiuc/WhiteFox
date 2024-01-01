
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv1 = torch.nn.Conv1d(1, 1, 1)
        self.conv3 = torch.nn.Conv3d(1, 1, 1)
    def forward(self, x):
        return self.conv1(x) + self.conv3(x)
# Inputs to the model
x1 = torch.randn(1, 1, 32)
