
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model1 = torch.nn.Linear(3, 6)
 
    def forward(self, x1):
        v1 = self.model1(x1)
        v2 = v1 * torch.clamp(v1+3, 0, 6)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3) 
