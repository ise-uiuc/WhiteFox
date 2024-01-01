
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(10,10)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 - x
        return v2

# Initializing the model and input tensor
m = Model()
__input__ = torch.rand((1,10))
