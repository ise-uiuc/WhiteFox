
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(32, 16) # One layer of a simple fully connected network
 
    def forward(self, x1, x2):
        v1 = self.fc(x1)
        v2 = v1.relu()
        v3 = torch.matmul(v2, x2)
        v4 = torch.cat([v3], dim=0)
        v5 = v4.view(-1,16)
        return v5

# Initializing the model
m = Model() 

# Inputs to the model
x1 = torch.randn(16, 32)
x2 = torch.randn(4, 32)
