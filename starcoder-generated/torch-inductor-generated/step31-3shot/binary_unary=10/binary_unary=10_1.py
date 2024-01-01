
class Model(torch.nn.Module):
    def __init__(self, in_features=25088, out_features=4096):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
 
    def forward(self, x1, x2):
        x = self.linear(x1)
        x = x + x2
        x = torch.relu(x)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 25088)
x2 = torch.randn(1, 4096)
