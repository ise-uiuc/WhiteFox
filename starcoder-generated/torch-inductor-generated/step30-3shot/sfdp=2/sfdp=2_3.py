
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x1, x2, x3):
        w1 = torch.matmul(x1, x2.transpose(-2, -1))
        w2 = w1.div(0.1)
        w3 = w2.softmax(dim=-1)
        w4 = torch.nn.functional.dropout(w3, p=0.1)
        w5 = torch.matmul(w4, x3)
        return w5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5, 20)
x2 = torch.randn(1, 10, 20)
x3 = torch.randn(1, 10, 20)
