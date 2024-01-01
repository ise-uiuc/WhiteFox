
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output
 
# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(bs, num_heads, q_len, dim_k)
k = torch.randn(bs, num_heads, k_len, dim_k)
v = torch.randn(bs, num_heads, v_len, dim_v)
