
import torchvision
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1, stride=1, padding=1)
        self.fc = torch.nn.Linear(2, 8)
        self.dropout = torch.nn.Dropout()
    def forward(self, x1, x2):
        v1 = torch.nn.functional.adaptive_avg_pool2d(self.conv(x1), size=None)
        v2 = torch.nn.functional.linear(v1, input=x2)
        v3 = self.fc(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 2)
