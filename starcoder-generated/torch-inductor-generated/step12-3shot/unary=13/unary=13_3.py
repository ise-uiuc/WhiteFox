
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, inputs):
        x1 = self.linear(inputs)
        t2 = torch.sigmoid(x1)
        t3 = x1 * t2
        return t3

# Initializing the model
m = Model()

# Inputs to the model
inputs = torch.randn(1, 3)
