
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 8)
 
    def forward(self, y1):
        w1 = self.fc(y1)
        w2 = w1 * 0.5
        w3 = w1 * 0.7071067811865476
        w4 = torch.erf(w3)
        w5 = w4 + 1
        w6 = w2 * w5
        return w6

# Initializing the model
m = Model()

# Inputs to the model
y1 = torch.randn(1, 8)
