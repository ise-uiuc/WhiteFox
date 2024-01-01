
class Model(torch.nn.Module):
    def __init__(self):
        super(Model):
            self.linear = torch.nn.Linear(5, 1)
           
    def forward(self, input):
        out = self.linear(input)
        out += other
        return out

# Initializing the model
m = Model()

# Inputs to the model
__input__ = torch.randn(1, 5)
