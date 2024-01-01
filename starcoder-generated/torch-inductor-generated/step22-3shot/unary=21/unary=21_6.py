
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__(p2, p1, p0)
        self.bn2 = torch.nn.BatchNorm2d(1024)
    def forward(self, x):
        x1 = self.bn2(x)
        x2 = self.relu(x1)
        x3 = self.conv2(x2)
    def forward(self, x544):
        x545 = self.conv(x544)
        x546 = x545.transpose(0, 2)
        x547 = self.relu(x546)
        x548 = self.conv1(x547)
        x549 = self.bn1(x548)
        return x549
# Inputs to the model
x544 = torch.randn(1, 3, 256, 256)
