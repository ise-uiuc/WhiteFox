
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 256)
 
    def forward(self, x1, x2=None):
        v1 = self.linear(x1)
        if not x2:
            v2 = v1 + v1
        else:
            v2 = v1 + x2
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.FloatTensor(11, 128)
x2 = None
