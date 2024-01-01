
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(6272, 1024)
        self.linear2 = torch.nn.Linear(1024, 512)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 - 5
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 6272)
