
class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def attention(self, query, key, value, dropout=0.3):
        q = query
        k = key
        v = value
        scale_factor = float(q.size(-1)) ** -0.5

        # Shape: (num_heads, batch_size, num_units, sequence_length)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = torch.nn.functional.softmax(scaled_qk, dim=-1)
        dropout_qk = F.dropout(softmax_qk, p=dropout)
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(16, 8, 64)
key = torch.rand(16, 8, 128)
value = torch.rand(16, 8, 128)
