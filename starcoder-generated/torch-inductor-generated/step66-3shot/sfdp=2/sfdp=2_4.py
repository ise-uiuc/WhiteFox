
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        return torch.matmul(x1, x2.transpose(-2, -1)).div(x3).softmax(dim=-1).matmul(x3).dropout(0.1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4, 24, 16)
x2 = torch.randn(1, 4, 16, 24)
x3 = torch.randn(1, 4, 16, 32)
