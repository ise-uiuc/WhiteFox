
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = torch.nn.Linear(19, 28)
 
    def forward(self, x):
        y = self.input_projection(x)
        return y

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(2, 19)
