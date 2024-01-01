
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(in_features=256, out_features=128, bias=True)
 
    def forward(self, x):
        v1 = self.fc(x)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(128, 256)
