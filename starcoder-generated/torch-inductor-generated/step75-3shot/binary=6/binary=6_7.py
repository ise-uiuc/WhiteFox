
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Embedding()
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - __other__
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randint(1, 16, (10,))
