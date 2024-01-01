
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 27, 1, stride=2, bias=True, padding=1)
    def forward(self, x1):
        t1 = self.conv1(x1)
        t2 = t1 - 0.1
        t3 = F.relu(t2)
        return t3
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
