
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(10240, 1024))
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = F.relu(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn
