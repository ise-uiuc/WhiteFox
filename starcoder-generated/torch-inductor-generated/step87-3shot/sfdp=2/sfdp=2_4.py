
class Model(torch.nn.Module):
    def __init__(self, num_heads, d_model, dropout_p, inv_scale_factor):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
 
        self.inv_scale_factor = inv_scale_factor
        self.dropout_p = dropout_p
 
        self.query = torch.nn.Parameter(torch.Tensor(num_heads, d_model, 1))
        self.key = torch.nn.Parameter(torch.Tensor(num_heads, d_model, 1))
        self.value = torch.nn.Parameter(torch.Tensor(num_heads, d_model, 1))
 
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, x1):
        b_s, seq_length, d_model = x1.size()
        q = x1 + self.query
        k = x1 + self.key
        v = x1 + self.value
 
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        out = dropout_qk.matmul(v)
        return out