
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.5
 
    def forward(self, q, k, v):
        d1 = torch.matmul(q, k.transpose(-2, -1))
        d2 = d1.mul(self.scale_factor)
        d3 = d2.softmax(dim=-1)
        d4 = torch.nn.functional.dropout(d3, p=self.dropout_p)
        output = d4.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 128, 64)
k = torch.randn(1, 128, 128)
v = torch.randn(1, 128, 128)
