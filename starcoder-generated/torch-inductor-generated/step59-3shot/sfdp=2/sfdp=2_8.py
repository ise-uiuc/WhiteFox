
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        o1 = torch.matmul(x1, x2.transpose(-2, -1))
        o2 = o1.div(0.12)
        o3 = torch.nn.functional.softmax(o2, dim=-1)
        o4 = torch.nn.functional.dropout(o3, 0.1)
        o5 = torch.matmul(o4, x3)
        return o5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 768, 56, 56)
x2 = torch.randn(1, 768, 56, 56)
x3 = torch.randn(1, 768, 56, 56)
