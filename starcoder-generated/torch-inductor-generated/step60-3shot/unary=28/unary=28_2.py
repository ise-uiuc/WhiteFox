
class Model(torch.nn.Module):
    def __init__(self, min_val=-1.0, max_val=1.0):
        super().__init__()
        self.linear = torch.nn.Linear(130, 133)
 
    def forward(self, data):
        data = torch.relu(data)
        x = self.linear(data)
        return x

# Initializing the model
m = Model()

# Inputs to the model
data = torch.randn(1, 130)
