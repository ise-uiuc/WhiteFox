
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8*8*8, 8)
 
    def forward(self, x2):
        v0 = x2.view(-1)
        v1 = self.linear(v0)
        v2 = v1.view(-1, 8, 8, 8)
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 8*8*8)
