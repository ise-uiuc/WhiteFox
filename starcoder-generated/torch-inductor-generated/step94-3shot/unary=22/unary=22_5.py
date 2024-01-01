
class Model(torch.nn.Module):
    # The model definition should be different from the previous one
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.tanh(v1)
        return v2

# Initializing the model, the size of model parameters should be different from the previous one
m = Model(3, 1)

# Inputs to the model
x1 = torch.randn(1, 3)
