
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=10, out_features=12)
 
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = v1 - 10
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
