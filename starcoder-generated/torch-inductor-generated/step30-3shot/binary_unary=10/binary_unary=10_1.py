
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn0 = torch.nn.BatchNorm2d(3)
        self.l1 = torch.nn.Conv2d(3, 4, 7, stride=2, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(4)
        self.l2 = torch.nn.Conv2d(4, 6, 3, stride=1, padding=0)
        self.bn2 = torch.nn.BatchNorm2d(6)
        self.l3 = torch.nn.Conv2d(6, 20, 1, stride=1, padding=0)
        self.relu = torch.nn.ReLU()
 
    def forward(self, x):
        v1 = self.l1(x)
        v2 = self.bn1(v1)
        v3 = self.l2(v2)
        v4 = self.bn2(v3)
        v5 = self.l3(v4)
        v6 = v5.mean([2, 3])
        v7 = self.relu(v6)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(100, 3, 224, 224)
