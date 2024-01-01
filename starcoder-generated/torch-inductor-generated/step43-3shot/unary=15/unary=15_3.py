
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(3, 64, (1,3), stride=1, padding=0), 
                                         torch.nn.BatchNorm2d(64),
                                         torch.nn.Flatten(),
                                         torch.nn.Linear(65536, 13))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
