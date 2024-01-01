
class Model(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features)
        self.relu = torch.nn.ReLU()
 
    def forward(self, x):
        v1 = self.fc(x)
        v2 = self.relu(v1)
        return v2

# Initializing the model
infeatures = 16
outfeatures = 4

m = Model(infeatures, outfeatures)

# Inputs to the model
x = torch.randn(1, infeatures)
