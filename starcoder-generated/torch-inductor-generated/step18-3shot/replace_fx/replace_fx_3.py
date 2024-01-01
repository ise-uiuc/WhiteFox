
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(32768, 1)
        self.dropout = nn.Dropout(0.5)
        self.bn = nn.BatchNorm2d(32)
        
    def forward(self, x):
        out = self.features(x)
        out = self.bn(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
# Inputs to the model
x = torch.randn(2,3,50,50)
