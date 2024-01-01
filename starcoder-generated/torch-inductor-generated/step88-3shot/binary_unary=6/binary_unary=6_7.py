
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(512, 2)
 
    def forward(self, x1):
        v1 = self.linear(input)
        v2 = v1 - 10
        v3 = F.relu(v2)
        return v3

# Initializing the model
m2 = Model()

# Inputs to the model
x1 = torch.randn(1, 512)
