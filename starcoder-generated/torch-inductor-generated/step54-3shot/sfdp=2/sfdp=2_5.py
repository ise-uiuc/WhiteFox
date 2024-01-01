
class Model(torch.nn.Module):
    def __init__(self, n_qkv, n_head, d_qk, d_v):
        super().__init__()
        self.dropout_p = 0.2
 
    def forward(self, query, key, value, mask_future_positions):
        inv_scale_factor = hidden_size ** -0.5
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p, training=self.training)
        output = dropout_qk.matmul(value)
        if mask_future_positions:
            # mask the future locations
            mask_shape = list(value.shape[:-1]) + [1]
            subsequent_mask = torch.triu(torch.ones(mask_shape), diagonal=1).to(dtype=value.dtype, device=value.device)
            output = output * subsequent_mask
        return output

# Initializing the model
n_batch = 1
n_qkv = 3
n_head = 2
d_qk = 1
d_v = 1
query = torch.randn(n_batch, n_qkv, n_head, d_qk)
key = torch.randn(n_batch, n_qkv, n_head, d_qk)
value = torch.randn(n_batch, n_qkv, n_head, d_v)
mask_future_positions = True
