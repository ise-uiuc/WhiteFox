
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.nn.functional.linear(x1, x2)
        v2 = v1 + x2
        return v2

# Initializing the model
m = Model()
x1 = torch.randn(1, 3)
x2 = torch.from_numpy(n1.array([[1, 2, 3]])).float()
