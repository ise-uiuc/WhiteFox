
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features = 3, out_features = 3)
        self.linear2 = torch.nn.Linear(in_features=8,  out_features=8)
 
    def forward(self, x1):
        m1 = self.linear1(x1)
        m2 = m1.clamp(min=0,max=6)
        m3 = m2.add(3)
        m4 = m3.mul(6)
        return m4

# Initializing the model
m = Model()

# Inputs to the model
