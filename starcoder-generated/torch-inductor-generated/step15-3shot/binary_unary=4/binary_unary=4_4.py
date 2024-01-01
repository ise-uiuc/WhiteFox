
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.__linear = torch.nn.Linear(8, 8)
 
    def forward(self, x1, x2 = torch.zeros((8, 8))):
        v1 = (self.__linear(torch.reshape(x1, (1, 1024))).squeeze(-1))
        return v1.matmul(x2).matmul(v1).sum(), v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 1, 1024)
x2 = torch.randn(2)
__output__, __residual_tensor__ = m(x1, x2)

