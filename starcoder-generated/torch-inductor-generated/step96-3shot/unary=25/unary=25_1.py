
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = torch.nn.Linear(10, 1)
 
    @property
    def linear(self):
        return self._linear

    @linear.setter
    def linear(self, value):
        self._linear = value

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * 0.01
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 10)
