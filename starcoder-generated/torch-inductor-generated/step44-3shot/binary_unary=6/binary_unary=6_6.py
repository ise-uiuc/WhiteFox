
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
        self.other = other
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model(torch.from_numpy(np.array([-5.3,-50.5,0,90.0])).float())

# Inputs to the model
x1 = torch.randn(1,8)
