
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 8)
 
    def forward(self, x1):
        o1 = self.fc(x1)
        o2 = o1 * 0.5
        o3 = o1 + (o1 * o1 * o1) * 0.044715
        o4 = o3 * 0.7978845608028654
        o5 = torch.tanh(o4)
        o6 = o5 + 1
        o7 = o2 * o6
        return o7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
