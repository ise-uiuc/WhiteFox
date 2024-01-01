
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 8)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = torch.pow(v1, 3)
        v3 = v1 * 0.5
        v4 = v3 + v2 * 0.044715
        v5 = v4 * 0.7978845608028654
        v6 = torch.tanh(v5)
        v7 = v6 + 1
        v8 = v2 * v7
        return v8

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
