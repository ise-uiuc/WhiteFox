
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = 1 / math.sqrt(q.size(-1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.09147719411243315)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = QueryKeyTransform()

# Inputs to the model
q = torch.randn(2, 3, 256)
k = torch.randn(2, 3, 400)
v = torch.randn(2, 3, 400)
