
class Model(torch.nn.Module):
    def __init__(self,
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=0.1):
        super().__init__()
        self.dropout_p = dropout_p
        self.scale_factor = query.shape[-1]**-.5 # Compute the inverse of the last dimension of the query and key tensors
        self.query = torch.nn.Parameter(query)
        self.key = torch.nn.Parameter(key)
        self.value = torch.nn.Parameter(value)
 
    def drop_rows_or_cols(self, t):
        r = (t * torch.rand((t.shape[0], 1))).floor()
        return t * (r / r.sum(keepdim=True,))
 
    def forward(self, x):
        dropout_qk = torch.nn.functional.dropout(
            (torch.matmul(self.query, self.key.transpose(-2, -1)) / self.scale_factor), self.dropout_p)
        return torch.matmul(self.drop_rows_or_cols(dropout_qk), self.value)

# Initializing the model
d_model, d_k, d_v = 64, 64, 64
batch_size, tgt_len, src_len = 3, 10, 20
x1 = torch.randn(batch_size, tgt_len, d_model)
x2 = torch.randn(batch_size, tgt_len, d_model)
x3 = torch.randn(batch_size, tgt_len, d_model)
