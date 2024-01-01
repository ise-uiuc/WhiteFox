
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 1)
 
    def forward(self, x1):
        w1 = self.linear(x1)
        w2 = w1 * 0.5
        w3 = w1 + w2 * w1 * w2 * 0.044715
        w4 = w3 * 0.7978845608028654
        w5 = torch.tanh(w4)
        w6 = w5 + 1
        w7 = w2 * w6
        return w7

# Initializing the model
q = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
