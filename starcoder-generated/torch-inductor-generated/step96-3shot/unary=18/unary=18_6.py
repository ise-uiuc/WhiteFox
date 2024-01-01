
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()     #
        self.features = nn.Sequential(      # 
            nn.Conv3d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm3d(num_features=32),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool3d(kernel_size=4, stride=1, padding=0),
        )

        self.classifier = nn.Sequential(  #
            nn.Linear(in_features=1600, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=20),
        )
    def forward(self, x):              #
        x = self.features(x)            #
        x = x.view(-1, 1600)             #
        x = self.classifier(x)          #
        #x = F.softmax(x)                # (1)
        return x
# Inputs to the model
x1 = torch.randn(1, 64, 80, 224, 224)
