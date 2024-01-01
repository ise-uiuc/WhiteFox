
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10, bias=False)
 
    def forward(self, x1, **kwargs):
        _list = [x1, kwargs['input.2']]
        v1 = self.linear(_list[0])
        v2 = v1 + _list[1]
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
x2 = torch.randn(1, 10)
