
class Model(torch.nn.Module):
    def __init__(self, query_channels, key_channels, num_hidden, dropout_p,
                 num_heads, scale_factor=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale_factor = scale_factor
        self.linear_qkv = torch.nn.Linear(query_channels, num_hidden * 3, bias=False)
        self.linear_o = torch.nn.Linear(num_hidden, query_channels, bias=False)
        self.dropout = torch.nn.Dropout(p=dropout_p)
 
    def forward(self, query, key, value):
        new_qkv_shape = query.size()[:-1] + \
                        (self.num_heads, 3 * self.scale_factor)
        qkv = self.linear_qkv(query).view(new_qkv_shape)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)
        qk = torch.matmul(q, k)
        inv_scale_factor = 1.0 / self.scale_factor
        scaled_qk = qk.mul_(inv_scale_factor)
        softmax_qk = torch.nn.functional.softmax(scaled_qk, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        final_output = output.transpose(-2, -3)
        concat_output = final_output.contiguous().view(output_shape)
        return self.linear_o(concat_output)

# Initializing the model
m = Model(query_channels, key_channels, num_hidden, dropout_p,
          num_heads, scale_factor)

# Initializing input tensors and outputs to the model
query = torch.randn(batch_size, query_channels, query_len)
key = torch.randn(batch_size, key_channels, key_len)
value = torch.randn(batch_size, value_channels, value_len)
