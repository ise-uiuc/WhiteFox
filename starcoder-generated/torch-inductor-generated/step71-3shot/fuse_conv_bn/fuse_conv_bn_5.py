
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # This is equivalent to the pattern that triggers the fusion optimization. 
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
        )
        # This can't be fused because the output of torch.nn.BatchNorm2d(64) is not used. 
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
        )
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(128, 10) # This layer will trigger recomputation as the output of this layer in block2 is not used. If this is changed to torch.nn.Linear(128,2), no recomputation will be triggered. 
    def forward(self, x):
        x = self.block1(x)
        x1 = self.block2(x)
        x2 = self.block3(x1)
        x2 = self.avgpool(x2)
        x2 = torch.flatten(x2, 1)
        x2 = self.fc(x2)
        return x2
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
