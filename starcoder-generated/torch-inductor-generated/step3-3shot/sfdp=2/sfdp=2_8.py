
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5):
        x6 = torch.matmul(x1, x2.transpose(-2, -1))
        x7 = x6 / x5
        x8 = torch.nn.functional.softmax(x7, dim=-1)
        x9 = torch.nn.functional.dropout(x8, p=x4)
        x10 = torch.matmul(x9, x3)
        return x10

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5, 1280)
x2 = torch.randn(1, 4, 256)
x3 = torch.randn(1, 4, 128)
x4 = torch.randn(1, 5, 128)
x5 = torch.randn(1)
