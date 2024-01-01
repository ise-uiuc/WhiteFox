
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
 
    def forward(self, t1, t2):
        v1 = torch.matmul(t1, t2.transpose(-2, -1))
        v2 = v1 / 0.1
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.1)
        return v4.matmul(t2)

# Initializing the model
m = Model()

# Inputs to the model
t1 = torch.randn(1, 3, 10)
t2 = torch.randn(1, 4, 10)
