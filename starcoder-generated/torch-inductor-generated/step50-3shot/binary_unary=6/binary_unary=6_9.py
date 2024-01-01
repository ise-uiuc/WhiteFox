
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 8)
 
    def forward(self, x1):
        return relu(self.linear(x1)-10)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
