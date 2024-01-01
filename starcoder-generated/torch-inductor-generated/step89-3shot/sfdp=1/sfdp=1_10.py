
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1)) 
        v2 = v1.div(1.5) 
        v3 = v2.softmax(dim=3) 
        v4 = torch.nn.functional.dropout(v3, p=0.2) 
        v5 = v4.matmul(x2) 
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 4, 2)
x2 = torch.rand(4, 3, 5)
