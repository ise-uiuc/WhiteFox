
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.tanh(v1)
        return v2

# Initializing the model and inputs to the model
m = Model()
