
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x2):
        v1 = self.linear(x2)
        v2 = v1 + torch.arange(8).unsqueeze(1).repeat(1, 3)
        v3 = (v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 3)
