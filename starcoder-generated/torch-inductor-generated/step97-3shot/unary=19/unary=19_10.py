
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 1)
 
    def forward(self, x2):
        v7 = self.linear(x2)
        v8 = torch.sigmoid(v7)
        return v8

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 128)
