
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=(64, 7, 7), stride=(1, 2, 2))
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.relu = torch.nn.ReLU(inplace=False)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(80, 64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.tanh = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(64, 1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.bn2(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 112, 112)
