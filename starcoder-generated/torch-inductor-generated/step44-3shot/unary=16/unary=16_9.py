
class Model(nn.Module):
    def __init__(self, i, o, k):
        super(Model, self).__init__()
        self.linear = nn.Linear(i, o, k)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1.relu()
        return v2

# Initializing the model
m = Model(3, 8, 100)

# Inputs to the model
x1 = torch.randn(1, 3, 100)
