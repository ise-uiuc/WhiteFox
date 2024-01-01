
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(8, 8, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.squeeze(v1) - torch.ones(8, 8)
        return v2
# Inputs to the model
x = torch.randn(1, 8, 640)
