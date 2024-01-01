
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 16, 5)
        self.conv1 = torch.nn.ConvTranspose2d(16, 4, 5)
        self.conv2 = torch.nn.ConvTranspose2d(4, 1, 2)
        
        self.fc = torch.nn.Linear(1024, 3072)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.bn3 = torch.nn.BatchNorm2d(4)
        self.bn4 = torch.nn.BatchNorm1d(1024)
        self.bn5 = torch.nn.BatchNorm1d(3072)
        
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn2(v1)
        v2 = torch.relu(v2)
        v3 = self.conv1(v2)
        v4 = self.bn3(v3)
        v4 = torch.relu(v4)
        v5 = self.conv2(v4)
        v6 = self.bn4(v5)
        v7 = v6.flatten(start_dim=1, end_dim=-1)
        v8 = self.fc(v7)
        v9 = self.bn5(v8)
        v10 = torch.relu(v9)
        v10 = v10.view(1, -1)
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
