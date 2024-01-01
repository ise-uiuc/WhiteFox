
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))    
        v2 = v1 / 1
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.8500700857028166)
        v5 = torch.matmul(v4, x3) 
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(64, 5, 100)
x2 = torch.randn(64, 100, 200)
x3 = torch.randn(64, 200, 400)
x4 = torch.randn(64, 5, 400)
