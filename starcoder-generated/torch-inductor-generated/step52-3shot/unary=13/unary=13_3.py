
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(5, 10, bias=False)
        self.sigmoid = torch.nn.Sigmoid()
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = self.sigmoid(v1)
        v3 = v1 * v2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(16, 5)
