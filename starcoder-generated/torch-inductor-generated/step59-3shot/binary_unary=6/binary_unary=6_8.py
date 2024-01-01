
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(28, 10)
 
    def forward(self, features):
        x = self.linear(features)
        x -= x.max()
        return x

# Initializing the model
m = Model()

# Inputs to the model
features = torch.randn(28, 20)
