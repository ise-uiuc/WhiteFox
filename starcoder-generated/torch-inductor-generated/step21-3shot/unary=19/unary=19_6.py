
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 10, bias=True)
 
    def forward(self, x1):
        v2 = self.linear(x1)
        v3 = torch.sigmoid(v2)
        return v3
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 16)
