
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 15, 5, stride=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 0.48
        v3 = F.relu(v2)
        return v3
# Inputs to the model
x1 = torch.rand(2, 256)
