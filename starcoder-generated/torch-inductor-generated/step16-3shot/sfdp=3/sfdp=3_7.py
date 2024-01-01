
class Model(torch.nn.Module):
    def forward(self, q, k, v, scale_factor=1.0, dropout_p=0.1):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk * (scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = F.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
batch_size, num_heads, qkv_size = 1, 8, 64
query = torch.randn(batch_size, num_heads, qkv_size)
key = torch.randn(batch_size, num_heads, qkv_size)
value = torch.randn(batch_size, num_heads, qkv_size)
