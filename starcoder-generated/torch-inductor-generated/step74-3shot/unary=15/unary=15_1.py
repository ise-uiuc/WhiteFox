
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(256*10*2, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 64)
        self.fc4 = torch.nn.Linear(64, 16)
        self.fc5 = torch.nn.Linear(16, 1)
    def forward(self, x1):
        v1 = x1.view(-1,256*10*2)
        v2 = torch.relu(self.fc1(v1))
        v3 = torch.relu(self.fc2(v2))
        v4 = torch.relu(self.fc3(v3))
        v5 = torch.relu(self.fc4(v4))
        v6 = self.fc5(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1,256,10,2)
