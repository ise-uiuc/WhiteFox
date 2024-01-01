
t1 = torch.randn(1, 3, 224, 224)
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(224, 128, 1, stride=2)
        self.maxpool = torch.nn.MaxPool2d(2, stride=2, padding=0)
    def forward(self, t1):
        v1 = self.conv(t1)
        v2 = v1 - torch.transpose(t1, 2, 1)
        v3 = F.relu(v2)
        v4 = v3 - self.maxpool(t1)
        v5 = F.relu(v4)
        return v5
# Inputs to the model
