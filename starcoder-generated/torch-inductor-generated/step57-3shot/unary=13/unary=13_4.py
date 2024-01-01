
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)
 
    def forward(self, input):
        v1 = self.linear(input)
        v2 = torch.sigmoid(v1) 
        v3 = v1 * v2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = 2 * torch.randn(1, 3)
