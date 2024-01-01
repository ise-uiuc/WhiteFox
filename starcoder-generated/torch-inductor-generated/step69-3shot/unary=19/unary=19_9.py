
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, v1):
        v2 = self.linear(v1)
        v3 = torch.sigmoid(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
v1 = torch.randn(1,8)
