
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        z1 = torch.matmul(x1, x1.transpose(-2, -1))
        p1 = z1.div(0.001)
        z2 = torch.matmul(p1, p1.transpose(-2, -1))
        o1 = z2.softmax(dim=-1)
        g1 = torch.nn.functional.dropout(o1, p=0.3, training=True)
        o2 = torch.matmul(g1, x2)
        return o2
 
# Initializing the model
model = Model()

# Inputs to the model
x1 = torch.randn(128, 64, 200)
x2 = torch.randn(128, 64, 500)
