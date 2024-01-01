
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(28, 27, 3, stride=1, padding=0)
        self.fc = torch.nn.Linear(7, 7)
    def forward(self, x1, other=None):
        v1 = self.conv(x1)
        v1 = F.max_pool2d(v1, 4)
        v1 = v1.view(-1, 7)
        if other == None:
            other = torch.randn(21)
        v2 = self.fc(v1) + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 28, 28, 28)
