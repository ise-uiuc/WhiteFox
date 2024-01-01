
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 4)
        self.conv = nn.Conv2d(4, 4, 3, padding=1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x, inplace=True)
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.conv(x)
        return x
# Inputs to the model
x = torch.randn(2, 4, 8, 8)
