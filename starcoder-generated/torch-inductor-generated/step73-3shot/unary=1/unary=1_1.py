
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(200, 300)
 
    def forward(self, x2):
        v11 = self.linear(x2)
        v12 = v11 * 0.5
        v13 = v11 * v11 * v11
        v13 = v13 * 0.044715
        v14 = v13 * 0.7978845608028654
        v14 = torch.tanh(v14)
        v14 = v14 + 1
        v15 = v12 * v14
        return v15

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(10, 200)
