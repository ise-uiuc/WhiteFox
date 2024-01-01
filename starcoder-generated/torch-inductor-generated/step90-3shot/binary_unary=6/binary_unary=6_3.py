
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8, bias=False)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 - other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
rng = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(123456789)).jumped())
x1 = torch.randn(1, 3, 64, 64)
other = rng.randint(10, size=(1,))[0]

