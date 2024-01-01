
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        k1 = torch.matmul(x1, x3.transpose(-2, -1))
        k2 = k1.div(0.30000001192092896)
        x4 = torch.nn.functional.softmax(k2, dim=-1)
        x5 = torch.nn.functional.dropout(x4, 0.2)
        x6 = torch.matmul(x5, x2)
        return x6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(16, 5, 5)
x2 = torch.randn(16, 5, 10)
x3 = torch.randn(5, 10)
