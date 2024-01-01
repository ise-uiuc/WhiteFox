
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.biasb = torch.nn.Parameter(torch.zeros(25920, 1, 1))
        self.scaleb = torch.nn.Parameter(torch.full((25920, 1, 1), 0.21))
    
    def forward(self, qt, k):
        a1 = torch.matmul(qt, k.transpose(-2, -1))
        a2 = a1 * self.scaleb
        a3 = a2.softmax(dim=-1)
        a4 = torch.nn.functional.dropout(a3, p=0.7978529607897845)
        a5 = torch.matmul(a4, self.biasb)
        a6 = a5 + qt
        return a6

# Initializing the model
m = Model()

# Inputs to the model
qt = torch.randn(1, 3, 64, 64)
k = torch.randn(1, 3, 64, 64)
