
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Number of in_features = 11
        # Number of out_features = 8
        self.fc = torch.nn.Linear(11, 8)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        # v1 is of size 1, 8. But x1 is of size 1, 11.
        v2 = torch.sub(v1, other)
        v3 = torch.nn.ReLU(v2, inplace=False)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 11)
