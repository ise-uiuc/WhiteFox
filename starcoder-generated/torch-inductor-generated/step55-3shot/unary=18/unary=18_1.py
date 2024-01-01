
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 64, 3),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, 3),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(128*12*12, 1024),
            torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(inplace=True)
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(inplace=True)
        )
        self.fc3 = torch.nn.Linear(512, 2)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(-1, 128*12*12)  # Flatten the input tensor first
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
