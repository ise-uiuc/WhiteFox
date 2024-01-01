
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 6)
 
    def forward(self, input1):
        _1 = self.linear(input1)
        _2 = torch.sigmoid(_1)
        _3 = _1 * _2
        return _3

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(1, 6)
