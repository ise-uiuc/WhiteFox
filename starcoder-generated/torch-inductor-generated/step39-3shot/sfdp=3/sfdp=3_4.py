
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.drop1 = torch.nn.Dropout(dropout_p)
        self.linear1 = torch.nn.Linear(10, 20)
        self.linear2 = torch.nn.Linear(20, 20)
    
    def forward(self, x1):
        x2 = torch.tanh(self.linear1(x1))
        x3 = torch.atan(self.linear2(x2))
        x4 = x3.clamp(1, 2)
        x5 = self.drop1(x4)
        x6 = torch.erf(x5)
        x7 = x6 + 1
        return x7

# Initializing the model
m = Model()

# Inputs to the model
__input__ = torch.randn(1, 10)
