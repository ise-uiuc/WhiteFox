
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(21, 28, 2)
        self.conv10 = torch.nn.Conv2d(9, 10, 1, stride=1, padding=10)
        self.conv5 = torch.nn.Conv2d(12, 14, 2, stride=1, padding=2)
        self.maxPool4 = torch.nn.Conv2d(13, 17, 2)
        self.linear = torch.nn.Linear(45, 51, 1)
    def forward(self, x40, x38, x39, x41):
        v43 = self.conv1(x40)
        v45 = x38 + v43
        v49 = v45 * 0.5
        v52 = self.conv10(v45)
        v44 = self.conv5(v45)
        v51 = v44 * 0.5
        v46 = self.maxPool4(v44)
        v55 = v46 * 0.5
        v54 = v46 * v46
        v48 = v54 * 0.5
        return v54
# Inputs to the model
x40 = torch.randn(1, 21, 26, 22)
x38 = torch.randn(1, 9, 6, 5)
x39 = torch.randn(1, 12, 19, 16)
x41 = torch.randn(1, 13, 19, 15)
