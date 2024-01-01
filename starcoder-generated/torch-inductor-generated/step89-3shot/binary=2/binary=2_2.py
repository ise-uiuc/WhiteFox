
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(5, 10, 4, stride=4, padding=0, groups=3)
    def forward(self, x17):
        v17 = self.conv(x17)
        v18 = v17 - torch.tensor([11, 21, 27, 32, 99, 8, 42, 27, 90, 65, 96, 72, 100, 85, 76, 52, 19, 93, 39]).float()
        return v18
# Inputs to the model
x17 = torch.randn(1, 5, 80)
