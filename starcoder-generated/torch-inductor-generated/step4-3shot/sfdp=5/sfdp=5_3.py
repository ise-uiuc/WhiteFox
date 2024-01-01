
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def _forward(self, q, k, v):
        a = torch.matmul(q, k.transpose(2, 3))
        s = a / (float(q.shape[-1]) ** 0.5)
        m = torch.nn.Softmax(dim=-1)(s + attn_mask)
        m = torch.nn.Dropout(dropout_p)(m)
        x = torch.matmul(m, v)
        return x

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 12, 256, 512)
k = torch.randn(1, 12, 256, 512)
v = torch.randn(1, 16, 256, 512)

