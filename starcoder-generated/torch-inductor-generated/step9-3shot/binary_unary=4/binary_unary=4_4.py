
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(11, 12)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + kwarg['other']
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 11)
kwarg={'other':torch.randn(1, 12)}
