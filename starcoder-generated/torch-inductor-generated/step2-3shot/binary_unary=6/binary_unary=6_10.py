
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 5, bias=False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 100
        v3 = F.relu(v2)
        return v3, v2

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(5)
