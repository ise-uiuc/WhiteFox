
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.BatchNorm2d(6),
            nn.Dropout(0.8),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 10, 5),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.features(x)
        v1 = torch.relu(x)
        return v1
# Inputs to the model
x = torch.randn(200,6,32,32)
