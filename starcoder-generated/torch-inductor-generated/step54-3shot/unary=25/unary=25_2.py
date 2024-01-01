
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)
 
    def forward(self, x1):
        w1 = self.linear(x1)
        w2 = w1 > 0
        w3 = w2 * 0.01
        w4 = torch.where(w2, w1, w3)
        return w4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
