
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = torch.nn.Linear(56 * 56, 256, bias=False)
        self.activ = torch.nn.ReLU()
 
    def forward(self, x):
        v1 = self.fc(x)
        v2 = self.activ(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 56 * 56)
