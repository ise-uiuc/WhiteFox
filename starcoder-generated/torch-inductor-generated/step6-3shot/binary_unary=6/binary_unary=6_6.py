
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 1)
 
    def forward(self, t1, t2):
        v1 = self.linear(t1)
        v2 = v1 - t2
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
t1 = torch.randn(4, 5)
t2 = torch.randn(4, 5)
