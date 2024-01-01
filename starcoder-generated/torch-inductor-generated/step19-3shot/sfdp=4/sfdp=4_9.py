
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x1, x2):
        qk = x1 @ x2
        qk = qk.transpose(-2, -1)
        return torch.softmax(qk, dim = -1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32, 128)
x2 = torch.randn(1, 128, 128)
