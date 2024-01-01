
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(3, 3, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv1d(3, 3, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv1d(3, 3, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv1d(3, 3, 1, stride=1, padding=1)
        self.conv5 = torch.nn.Conv1d(3, 3, 1, stride=1, padding=1)
        self.conv6 = torch.nn.Conv1d(3, 3, 1, stride=1, padding=1)
        self.conv7 = torch.nn.Conv1d(3, 3, 1, stride=1, padding=1)
        self.conv8 = torch.nn.Conv1d(3, 3, 1, stride=1, padding=1)
        self.conv9 = torch.nn.Conv1d(3, 3, 1, stride=1, padding=1)
        self.conv10 = torch.nn.Conv1d(3, 3, 1, stride=1, padding=1)
        self.conv11 = torch.nn.Conv1d(3, 3, 1, stride=1, padding=1)
        self.conv12 = torch.nn.Conv1d(3, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        x1 = 13.37 + x1
        x2 = self.conv1(x1)
        x3 = 13.37 + self.conv2(x2)
        x4 = 13.37 + self.conv3(x3)
        x5 = 13.37 + self.conv4(x4)
        x6 = 13.37 + self.conv5(x5) * x2
        x7 = 13.37 + self.conv6(x6) * x3
        x8 = 13.37 + self.conv7(x7) * x4
        x9 = 13.37 + self.conv8(x8) * x5
        x10 = self.conv9(x9) * x6
        x11 = self.conv10(x10) * x7
        x12 = self.conv11(x11) * x8
        x13 = self.conv12(x12) * x9
        return x13
# Inputs to the model
x1 = torch.randn(1, 3, 256)
