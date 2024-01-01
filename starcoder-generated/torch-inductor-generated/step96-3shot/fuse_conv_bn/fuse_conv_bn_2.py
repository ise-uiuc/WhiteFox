
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Conv2d(6, 6, 1, bias=True)
        self.layer2 = torch.nn.Conv2d(6, 6, 1, stride=2, padding=1, bias=True)
        self.layer3 = torch.nn.Conv2d(6, 6, 3, padding=1, bias=True)
        self.layer4 = torch.nn.Conv2d(6, 12, 1, bias=True)
        self.layer5 = torch.nn.Conv2d(12, 12, 1, stride=2, padding=1, bias=True)
        self.layer6 = torch.nn.Conv2d(12, 3, 3, padding=1, bias=True)
        self.fc1 = torch.nn.Linear(36, 16, bias=True)
        self.fc2 = torch.nn.Linear(36, 3, bias=True)
    def forward(self, x3):
        s3 = self.layer1(x3)
        s3 = self.layer2(s3)
        s3 = self.layer3(s3)
        s3 = self.layer4(s3)
        s3 = self.layer5(s3)
        s3 = self.layer6(s3)
        s3 = s3.view(s3.size(0), -1)
        s3 = self.fc1(s3)
        s3 = self.fc2(s3)
        x3 = s3 + s3
# Inputs to the model
x3 = torch.randn(1, 3, 6, 6)
