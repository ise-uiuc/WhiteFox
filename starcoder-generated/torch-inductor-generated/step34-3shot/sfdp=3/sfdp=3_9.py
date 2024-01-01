
class Model(torch.nn.Module):
    def __init__(self, dim0, dim1, dim2):
        super().__init__() 
        self.dim0 = dim0
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x0, x1):
        s1 = torch.flatten(x0, start_dim=1)
        s2 = torch.flatten(x1, start_dim=1).transpose(0,1)
        v3 = torch.matmul(s1, s2) # query x key
        v4 = v3 * 0.5 # scale dot product
        v5 = torch.nn.functional.softmax(v4, dim=-1) # softmax
        v6 = torch.nn.functional.dropout(v5, p=0.2) # drop out
        v7 = torch.matmul(v6, s2.transpose(0, 1)) # attantion
        return v7

# Initializing the model
m = Model(128, 1, 70)

# Inputs to the model
x0 = torch.randn(1, 128, 1, 70)
x1 = torch.randn(1, 128, 70, 1)
