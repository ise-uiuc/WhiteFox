
class Model(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
    
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(0.10000000000000001)
        v3 = v2.softmax(dim=3)
        v4 = torch.nn.functional.dropout(v3, p=0.10000000149011612)
        v5 = v4.matmul(x2)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 3, 4, 5)
x2 = torch.randn(2, 3, 5, 7)
