
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5):
        x1 = torch.matmul(x1, x2.transpose(-2, -1))
        x2 = x1.div(x3)
        x3 = torch.nn.functional.softmax(x2, dim=-1)
        x4 = torch.nn.functional.dropout(x3, p=x4)
        x5 = torch.matmul(x4, x5)
        return x5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 4)
x2 = torch.randn(3, 4)
x3 = torch.randn(1)
x4 = torch.randn(5)
x5 = torch.randn(5, 4)
