
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v, x):
        qk = torch.matmul(q, k.transpose(-2, -1))
        return qk

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 64, 48)
k = torch.randn(1, 48, 36)
v = torch.randn(1, 36, 57)
x = torch.randn(1, 1, 48, 36)
