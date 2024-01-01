
class Model(torch.nn.Module):
    def __init__(self, m1):
        super().__init__()
        self.m1 = m1
 
    def forward(self, x1):
        v1 = self.m1(x1)
        o1 = v1 - other
        o2 = F.relu(o1)
        return o2

# Initializing the model
m1 = Model1()
m = Model(m1)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
