
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 8)
 
    def forward(self, x):
        v1 = self.fc(x)
        v2 = v1 * 0.5
        v3 = torch.tanh(0.7978845608028654 * (v1 + 0.044715 * v1 * v1 ))
        v4 = v2 * v3
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3)
