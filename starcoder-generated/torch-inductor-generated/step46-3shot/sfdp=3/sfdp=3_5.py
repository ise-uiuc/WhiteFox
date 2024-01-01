
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        p1 = torch.matmul(x1, x2.transpose(-2, -1))
        r1 = p1.mul(0.125)
        s1 = r1.softmax(dim=-1)
        d1 = torch.nn.functional.dropout(s1, p=0.125)
        output = d1.matmul(x2)
        return output

# Initializing the model using the input tensors
m = Model()
x1 = torch.randn(1, 16, 512)
x2 = torch.randn(1, 32, 512)
output = m(x1, x2)

