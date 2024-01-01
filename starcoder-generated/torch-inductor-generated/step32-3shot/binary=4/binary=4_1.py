
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 256)
 
    def forward(self, x, other):
        v1 = self.linear1(x)
        v2 = v1 + other
        return v2

# Initializing the model and a tensor
m = Model()
other = torch.tensor([[1., 2.]])
x = torch.randn(1, 10)
