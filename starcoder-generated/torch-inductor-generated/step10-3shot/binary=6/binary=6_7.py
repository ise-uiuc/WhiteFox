
class Model(torch.nn.Module, Exporter):
    def __init__(self, other):
        super().__init__()
        self.linear = Linear(16 * 5 * 5, 10)
        self.other = other
 
    @Export(method='forward')
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 - self.other
        return v2
 
# Initializing the model
m = Model(1)

# Inputs to the model
x = torch.randn(1, 1, 28, 28)
