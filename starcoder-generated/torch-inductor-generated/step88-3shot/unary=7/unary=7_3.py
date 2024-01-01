
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
 
    def forward(self, x1):
        w1 = self.linear(x1)
        w2 = F.relu(w1)
        w3 = w2 * torch.clamp(F.relu(w1+3), 0, 6)
        w4 = w3 / 6
        return w4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
