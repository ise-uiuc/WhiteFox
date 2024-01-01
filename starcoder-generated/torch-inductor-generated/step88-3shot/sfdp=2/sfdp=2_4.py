
class Model(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.wq = torch.nn.Linear(d_model, d_model)
        self.wk = torch.nn.Linear(d_model, d_model)
        self.wv = torch.nn.Linear(d_model, d_model)
  
    def forward(self, query, key, value, mask=None):
        q = self.wq(query)
        k = self.wk(key)
        v = self.wv(value)
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = 1.0 / np.sqrt(self.d_model)
        scaled_qk = qk * inv_scale_factor
        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.repeat(1, self.num_heads, 1, 1)
            scaled_qk = scaled_qk.masked_fill(mask==0, float('-1e20'))
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        return dropout_qk.matmul(v)
  
# Initializing the model
m = Model(d_model, num_heads)

# Inputs to the model
query = torch.randn(batch, seq_length, d_model)
key = torch.randn(batch, seq_length, d_model)
value = torch.randn(batch, seq_length, d_model)
