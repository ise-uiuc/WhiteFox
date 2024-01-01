
class Model(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=100, out_features=32)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
model = Model()
 
# Inputs to the model
x1 = torch.rand(1, 100)
x2 = torch.ones(1, 32)
