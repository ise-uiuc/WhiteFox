
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        s = x1.size(-1)
        q = x1.reshape(-1, s, 1)
        k = x2.reshape(-1, 1, s)
        v = x2.reshape(-1, s, 1)
        qk = torch.matmul(q, k)
        inv_scale_factor = 1.0 / math.sqrt(s)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 3, 4)
x2 = torch.randn(2, 3, 4)
