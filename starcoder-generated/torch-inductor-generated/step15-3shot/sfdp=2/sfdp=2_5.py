
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 / 32
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.1, training=False)
        v5 = torch.matmul(v4, x2)
        return v5
 
m = Model()
 
# Inputs to the model
x1 = torch.randn(1, 32, 512)
x2 = torch.randn(1, 512, 64)
