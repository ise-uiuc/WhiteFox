
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, v1, v2):
        v3 = torch.matmul(v1, v2.transpose(-2, -1))
        v4 = v3.div(9.765625)
        v5 = v4.softmax(dim=-1)
        v6 = torch.nn.functional.dropout(v5, p=0.3062484076194763)
        v7 = torch.matmul(v6, v2)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
v1 = torch.randn(1, 16, 256)
v2 = torch.randn(1, 2, 256)
