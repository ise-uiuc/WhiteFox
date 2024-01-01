
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x3, x4):
        z5 = torch.matmul(x3, x4.transpose(-2, -1))
        z6 = z5 * 0.5
        z7 = z6.softmax(dim=-1)
        z8 = torch.nn.functional.dropout(z7, p=0.0)
        z9 = z8.matmul(x4)
        z10 = torch.matmul(x4, z9.transpose(-2, -1))
        return z10

# Initializing the model
m = Model()

# Inputs to the model
x3 = torch.randn(1, 512, 28, 28)
x4 = torch.randn(1, 512, 28, 28)
