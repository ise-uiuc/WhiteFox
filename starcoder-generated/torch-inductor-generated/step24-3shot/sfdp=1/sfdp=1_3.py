
class Model(torch.nn.Module):
    def __init__(self):
        pass
 
    def forward(self, x1, x2):
        x3 = x2.transpose(-2, -1)
        v1 = torch.matmul(x1, x3)
        v2 = v1 / 0.06928203230275509
        v3 = v2.softmax(-1)
        v4 = F.dropout(v3, p=0.05)
        v5 = torch.matmul(v4, x2)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.ones(1, 3, 5)
x2 = torch.ones(1, 5, 2)
