
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.fc = torch.nn.Linear(7, 7)
    def forward(self, x1):
        m1 = self.conv(x1)
        m2 = torch.relu(m1)
        m3 = m2.permute((0, 2, 3, 1))
        m4 = self.fc(m3)
        return m4
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
