
# Note that the values of the model parameters in __init__() is completely different from the the previous one
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.fc = torch.nn.Linear(in_features=576, out_features=10, bias=False)
    
    def forward(self, x):
        v1 = self.conv(x)
        # Note that we now add a new parameter "other" for the add operator
        v2 = torch.relu(v1 + 0.1) 
        v3 = self.fc(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
