
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 48, 1, stride=1, padding=1)
        self.fc = torch.nn.Linear(48*14*15, 1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v1 = v1.view(v1.size(0), -1)
        v1 = self.fc(v1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
