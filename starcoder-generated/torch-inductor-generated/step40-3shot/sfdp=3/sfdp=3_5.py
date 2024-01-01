
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(query, key, value, scale_factor, dropout_p):
        v1 = torch.matmul(query, key.transpose(-2, -1))
        v2 = v1 * scale_factor
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=dropout_p)
        output = v4.matmul(value)
        return output

# Initializing the model
batch_size, seq_len, dim_size = (1, 3, 128+128)
dropout_p = 0.1
query = torch.randn(batch_size, seq_len, dim_size)
key = torch.randn(batch_size, seq_len, dim_size)
value = torch.randn(batch_size, seq_len, dim_size)
scale_factor = torch.sigmoid(torch.randn(1))
