
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8, bias=False)
        self.other = other
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model(torch.FloatTensor([1,2,3,4,5,6,7,8]))

# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
