
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 8)
 
    def forward(self, x17):
        v17 = self.linear(x17)
        v18 = v17 * 0.5
        v19 = v17 * v17 * v17 * 0.044715
        v20 = v19 + v17
        v21 = v20 * 0.7978845608028654
        v22 = torch.tanh(v21)
        v23 = v22 + 1
        v24 = v18 * v23
        return v24

# Initializing the model
m = Model()

# Inputs to the model
x17 = torch.randn(1, 4)
