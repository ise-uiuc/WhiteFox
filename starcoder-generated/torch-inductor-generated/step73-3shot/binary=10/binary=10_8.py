
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        features = model.features
        self.features = torch.nn.Sequential(
            features.conv0,
            features.norm0,
            features.relu0,
            features.pool0,
            features.res2,
            features.res3,
            features.res4,
            features.res5,
        )
        self.fc = features.classifier
        
    def forward(self, x_1):
        x_2 = self.features(x_1)
        v1 = torch.flatten(x_2, 1)
        v2 = self.fc(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 300, 300)
