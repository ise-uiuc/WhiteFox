
class Model(torch.nn.Module):
    def __init__(self):
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.fc = torch.nn.Linear(16 * 46 * 46, 10)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1.view(x.size(0), 16 * 46 * 46)
        v3 = self.fc(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
