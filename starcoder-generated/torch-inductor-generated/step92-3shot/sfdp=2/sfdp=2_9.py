
class DropoutLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.dropout = torch.nn.Dropout(dropout)
 
    def forward(self, x1):
        x2 = self.linear(x1)
        x3 = self.dropout(x2)
        return x3
 
class Attention(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale_factor = dim**-0.5
        self.q = DropoutLinear(dim, dim, dropout=0.5)
        self.k = DropoutLinear(dim, dim, dropout=0.5)
        self.v = DropoutLinear(dim, dim, dropout=0.5)
 
    def forward(self, x1, x2):
        q = self.q(x1)
        k = self.k(x1)
        v = self.v(x2)
        return qk.div(self.scale_factor)
 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 128
        self.attention = Attention(self.dim)
 
    def forward(self, x1):
        qk = self.attention(x1, x2)
        output = torch.matmul(qk, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, self.dim)
x2 = torch.randn(1, self.dim)
