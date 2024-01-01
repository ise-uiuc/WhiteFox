
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.randn(28, 256)
        self.k = torch.randn(28, 256)
        self.v = torch.randn(28, 256)
        
    def forward(self, x):
        x = torch.matmul(self.q, self.k.transpose(-2, -1))
        x = x * 0.5
        x = x.softmax(dim=1)
        x = torch.nn.functional.dropout(x, 0.125)
        x = torch.matmul(x, self.v)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 28, 256)
