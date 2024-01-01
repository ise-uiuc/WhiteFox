
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = torch.nn.Linear(1, 3)
 
    def forward(self, x1):
        v1 = self.input_transform(x1)
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1)
