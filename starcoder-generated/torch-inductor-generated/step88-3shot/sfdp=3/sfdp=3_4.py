
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v, scale_factor, dropout_p):
        m1 = torch.matmul(q, k.transpose(-2, -1))
        m2 = m1 * scale_factor
        m3 = m2.softmax(dim=-1)
        m4 = torch.nn.functional.dropout(m3, p=dropout_p)
        m5 = torch.matmul(m4, v)
        return m5

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 512, 1, 1, 1024)
k = torch.randn(1, 256, 1, 1, 1024)
v = torch.randn(1, 256, 1, 1, 1024)
