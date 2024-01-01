
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(2, 6, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(6, 8, 3, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8, 9, 3, stride=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(9, 10, 1, stride=1, padding=0),
            torch.nn.ReLU(),            
        )
    def forward(self, x):
        v1 = self.conv(x)
        v2 = 3 - v1
        return v2
# Inputs to the model
x = torch.randn(3, 2, 128, 128)
