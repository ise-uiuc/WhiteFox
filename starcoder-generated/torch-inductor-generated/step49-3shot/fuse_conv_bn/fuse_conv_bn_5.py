
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_1 = torch.nn.Linear(16, 16)
        self.bn1 = torch.nn.BatchNorm1d(16)
        self.relu_1 = torch.nn.ReLU()
        self.fc_2 = torch.nn.Linear(16, 16)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.relu_2 = torch.nn.ReLU()
        self.fc_3 = torch.nn.Linear(16, 10)
    def forward(self, x):
        x = self.fc_1(x)
        x = self.bn1(x)
        x = self.relu_1(x)
        x = self.fc_2(x)
        x = self.bn2(x)
        x = self.relu_2(x)
        x = self.fc_3(x)
        return x
# Inputs to the model 
x = torch.randn(16, 16)
