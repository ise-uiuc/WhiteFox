
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        q1 = torch.matmul(x1, x2.transpose(-2, -1))
        scale_factor = 1. / math.sqrt(q1.shape[-1])
        v2 = q1.div(scale_factor)
        v3 = torch.exp(v2)
        v4 = torch.nn.functional.dropout(v3, 0.40395)
        v5 = torch.matmul(v4, x2)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5, 7, 11)
x2 = torch.randn(8, 11, 20)
