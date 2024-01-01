
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)
 
    def forward(self, x1):
        z1 = self.linear(x1)
        z2 = torch.sigmoid(z1)
        return z2


# Initializing the model
model = Model()

# Inputs to the model
