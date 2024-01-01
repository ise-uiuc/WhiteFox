
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.25
  
    def forward(self, query, key, value):
        inv_scale_factor = torch.Tensor.new_tensor([1. / sqrt(query.shape[-1])])
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(batch_size, num_heads, seq_len, num_dims)
key = torch.randn(batch_size, num_heads, seq_len, num_dims)
value = torch.randn(batch_size, num_heads, seq_len, num_dims)
