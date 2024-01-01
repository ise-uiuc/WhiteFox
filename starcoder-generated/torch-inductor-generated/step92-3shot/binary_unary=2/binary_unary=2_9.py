
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 7, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.max(v1, -1)[0]
        v3 = torch.reshape(v2, [1, -1])
        return v3
# Inputs to the model
x1 = torch.ones(1, 3, 16, 16)
