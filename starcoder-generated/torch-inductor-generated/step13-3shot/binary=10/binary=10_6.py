
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(5, 2)
        self._other = torch.from_numpy(other)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self._other
        return v2

# Initializing the model
model = Model(np.random.rand(1, 5))
x1 = torch.randn(1, 1, 5)
