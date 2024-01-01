
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        q = x1
        k = x1
        v = x2
        m1 = torch.matmul(q, k.transpose(-2, -1))
        scale = float(m1.shape[-1]) ** -0.5
        m2 = m1 * scale
        out = torch.nn.functional.softmax(m2, dim=-1)
        dropout_res = torch.nn.functional.dropout(out, p=0)
        res = torch.matmul(dropout_res, v)
        return res

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 128)
x2 = torch.randn(8, 128, 10)
