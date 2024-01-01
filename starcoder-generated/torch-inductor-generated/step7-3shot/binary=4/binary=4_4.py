
class MyModule(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(3, 6)
        self.other = torch.from_numpy(other)
 
    def forward(self, x):
        return self.linear(x) + self.other

m = MyModule(other=np.random.uniform(0, 1, size=(6)).astype(np.float32))

# Inputs to the model
x = torch.randn(1, 3)
