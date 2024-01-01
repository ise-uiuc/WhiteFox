
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Attention()
 
    def forward(self, x1, x2):
        q, k, v = x1, x2, x2
        _, _, v = self.attention(q, k, v)
        return v

class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v):
        inv_scale_factor = k.shape[-1] ** -0.25
        dropout_p = 0.1
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output, None, None

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4, 16)
x2 = torch.randn(1, 4, 16)
