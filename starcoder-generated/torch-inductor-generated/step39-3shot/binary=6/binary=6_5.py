
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(245, 8)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 - np.pi
        return v2

# Initializing the model
m = Model()

x = torch.randn(2, 245)
