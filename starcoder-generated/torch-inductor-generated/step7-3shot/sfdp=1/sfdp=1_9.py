
class Model(torch.nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
 
    def forward(self, q, k, v, inv_scale_factor):
        attn = torch.matmul(q, k.transpose(-2, -1))
        scaled_attn = attn.div(inv_scale_factor)
        softmax_attn = scaled_attn.softmax(dim=-1)
        dropout_attn = torch.nn.functional.dropout(softmax_attn, p=self.dropout_p)
        output = torch.matmul(dropout_attn, v)
        return output

# Initializing the model
m = Model(dropout_p=0.1)

# Inputs to the model
q = torch.randn(1, 6, 64)
k = torch.randn(1, 20, 64)
v = torch.randn(1, 20, 64)
inv_scale_factor = 1 / math.sqrt(k.size(-1))
