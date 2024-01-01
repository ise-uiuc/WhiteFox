
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.Linear(224, 11)
 
    def forward(self, x1):
        v1 = self.norm(x1)
        return v1.squeeze()

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 224)
