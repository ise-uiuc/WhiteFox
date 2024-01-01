
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.linear1 = torch.nn.Linear(128, 64)
        self.linear2 = torch.nn.Linear(64, 32)
        self.linear3 = torch.nn.Linear(32, 8)
 
    def forward(self, x):
        b = self.flatten(b)
        l1 = self.linear1(b)
        l2 = self.linear2(l1)
        l3 = self.linear3(l2)
        o = torch.cat([l1, l2, l3], dim=1)
        return o

# Initializing the model
m = Model()

# Inputs to the model
b = torch.randn(1, 1, 40960)
x = torch.randn(1, 3, 64, 64)
