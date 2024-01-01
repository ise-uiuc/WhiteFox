
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 12, 1, stride=1, padding=0)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=19, stride=19, padding=0)
        self.fc = torch.nn.Linear(12, 109)
    def forward(self, x1):
        v97 = self.conv(x1)
        v98 = self.conv(x1)
        v99 = self.conv(x1)
        v100 = self.avg_pool(v99)
        v101 = self.avg_pool(v97)
        v102 = self.conv(x1)
        v103 = v101 * v102
        v104 = self.avg_pool(v97)
        v105 = self.conv(x1)
        v106 = self.avg_pool(v97)
        v107 = self.conv(x1)
        v108 = v104 + v105 + v106 + v107
        v109 = self.fc(v108)
        v110 = self.conv(x1)
        v111 = v97 + v98 + v99 + v100 + v103 + v109 + v110
        return v111
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
