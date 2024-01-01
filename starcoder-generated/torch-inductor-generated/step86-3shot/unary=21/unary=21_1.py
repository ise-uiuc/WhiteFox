
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(576, 256, 1, stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(256)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(256, 896, 1, stride=1, padding=0)
        self.bn2 = torch.nn.BatchNorm2d(896)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(896, 1000, 1, stride=1, padding=0)
        self.bn3 = torch.nn.BatchNorm2d(1000)
        self.relu3 = torch.nn.ReLU()
        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = torch.nn.Linear(1000, 8271)
    def forward(self, input0):
        r1 = self.conv1(input0)
        r2 = self.bn1(r1)
        r3 = self.relu1(r2)
        r4 = self.conv2(r3)
        r5 = self.bn2(r4)
        r6 = self.relu2(r5)
        r7 = self.conv3(r6)
        r8 = self.bn3(r7)
        r9 = self.relu3(r8)
        r10 = self.gap(r9)
        r11 = torch.flatten(r10, 1)
        r12 = self.fc1(r11)
        return r12
# Inputs to the model
input_0 = torch.randn(1, 576, 7, 7)
