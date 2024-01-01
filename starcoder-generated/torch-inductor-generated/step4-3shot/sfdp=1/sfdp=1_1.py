
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v4 = v1.div(10.0)
        v2 = v4.softmax(dim=-1)
        v5 = torch.nn.functional.dropout(v2, p=0.5)
        v6 = torch.matmul(v5, x3)
        
        return v6
    
# Initializing the mode
m = Model()

# Inputs
x1 = torch.randn(4, 6, 5)
x2 = torch.randn(4, 5, 7)
x3 = torch.randn(4, 7, 9)
