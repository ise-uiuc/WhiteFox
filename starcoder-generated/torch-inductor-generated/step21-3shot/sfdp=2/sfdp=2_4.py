
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x1, x2):
        qk = (x1 * x2).sum(-1).softmax(dim=-1)
        output = qk.matmul(x2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 48)
x2 = torch.randn(65, 48)
