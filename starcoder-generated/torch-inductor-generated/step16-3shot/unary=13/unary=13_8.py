
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        n_classes = 10
        n_features = 50
        mhidden = 100
        self.fc1 = torch.nn.Linear(n_features, mhidden)
        self.fc2 = torch.nn.Linear(mhidden, n_classes)
 
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.fc2(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(128, 50)
