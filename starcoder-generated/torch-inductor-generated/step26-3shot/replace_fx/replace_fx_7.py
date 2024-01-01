

class Dropout(torch.nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.dropout = torch.nn.dropout(p=p, inplace=inplace)
    def forward(self, x):
        return self.dropout(x)
    
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = Dropout(p=0.5)
        
    def forward(self, x1):
        x2 = self.dropout(x1)
        x3 = torch.rand_like(x1)
        return x3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
