
class Model(torch.nn.Module):
    def __init__(self): # An extra parameter
        super().__init__()
        self.linear = torch.nn.Linear(16, 1024)
 
        self.other = 2.0
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.other
        v3 = tanh(v2)
        return v3
 
