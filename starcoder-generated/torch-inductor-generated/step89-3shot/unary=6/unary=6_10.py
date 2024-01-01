
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_h = torch.nn.Conv2d(2, 5, 1, stride=1, padding=1)
        self.conv1_w = torch.nn.Conv2d(2, 5, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1_h(x1)
        v2 = self.conv1_w(torch.transpose(x1, 2, 3))
        v3 = torch.unsqueeze(v1 + v2, 1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
