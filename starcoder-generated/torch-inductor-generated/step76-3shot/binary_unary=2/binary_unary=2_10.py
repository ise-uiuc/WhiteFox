
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 128, 3, padding=(1, 2))
        self.conv2 = torch.nn.Conv2d(128, 8, 1)
    def forward(self, x0):
        v0 = self.conv1(x0)
        tmp = v0 - 1.3622
        tmp = F.relu(tmp)
        tmp = self.conv2(tmp)
        tmp = tmp - 38.9350
        return tmp
# Inputs to the model
x0 = torch.randn(1, 3, 30, 30)
