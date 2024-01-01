
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.nn.Linear(1024, 3072)
        self.t2 = torch.nn.Linear(1024, 3072)
 
    def forward(self, x1):
        v1 = self.t1(x1)
        v2 = v1 + self.t2.weight
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 1024)
