
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        x3 = torch.matmul(x1, x2.transpose(-2, -1))
        x4 = x3 * 1.0
        x5 = x4.softmax(dim=-1)
        x6 = torch.nn.functional.dropout(x5, p=0.1)
        v1 = torch.matmul(x6, x2)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(7, 12, 32)
x2 = torch.randn(7, 6, 48)
