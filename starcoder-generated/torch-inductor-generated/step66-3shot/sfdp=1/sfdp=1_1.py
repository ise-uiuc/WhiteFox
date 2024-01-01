
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def scaled_dot_product_attention(self, query, key, value, dropout, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout)
        output = dropout_qk.matmul(value)
        return output
 
    def forward(self, x1, x2, x3, x4):
        dropout = 0.8
        inv_scale_factor = 1 / math.sqrt(x2.size(-1))
        scaled_attention = self.scaled_dot_product_attention(x1, x2, x3, dropout, inv_scale_factor)
        v1 = scaled_attention + x4
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4, 30, 64)
x2 = torch.randn(1, 4, 30, 64)
x3 = torch.randn(1, 4, 30, 64)
x4 = torch.randn(1, 4, 30, 64)
__output_m__ = m(x1, x2, x3, x4)

