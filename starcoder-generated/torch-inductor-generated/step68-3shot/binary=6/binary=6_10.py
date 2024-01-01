
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 64)
 
    def forward(self, x1):
        t1 = self.linear(x1)
        t2 = t1 - float_t1
        return t2

# Initializing the model
m = Model()

# FloatTensor of an example
float_t1 = torch.tensor([10.0])

# Inputs to the model
x1 = torch.randn(1, 3)
