
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 32)
 
    def forward(self, x1, w2):
        v1 = self.linear(x1)
        v2 = v1 - w2
        return v2

# Initializing the model
m = Model()
 
# Inputs to the model
x1 = torch.randn(1, 8)
w2 = torch.randn(1, 32)
