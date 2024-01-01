
class Attention(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, query, key, value):
        dropout_p = 0.1
        query = query.div(math.sqrt(self.d))
        key = key.div(math.sqrt(self.d))
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = math.sqrt(self.d)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output
 
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d, num_heads):
        super().__init__()
        self.d = d
        self.num_heads = num_heads
        self.attns = []
        for i in range(num_heads):
            self.attns.append(Attention(d))
        self.attns = torch.nn.ModuleList(self.attns)
 
    def forward(self, query, key, value):
        # Reshape the tensors from BxTxD to Bh*TxD
        batch_size, time_steps, dims = query.shape
        query = query.view(batch_size, time_steps, self.num_heads, -1).transpose(1, 2).reshape(-1, time_steps, dims)
        key = key.view(batch_size, time_steps, self.num_heads, -1).transpose(1, 2).reshape(-1, time_steps, dims)
        value = value.view(batch_size, time_steps, self.num_heads, -1).transpose(1, 2).reshape(-1, time_steps, dims)
        attns = [attn(query, key, value) for attn in self.attns]
        output = torch.stack(attns, axis=2).reshape(batch_size, self.num_heads * time_steps, -1)
        # Reshape the tensors back into BxTxHxD
        output = output.transpose(1, 2).view(batch_size, self.num_heads * time_steps, -1).view(batch_size, time_steps, -1)
        return output

# Initializing a MultiHeadAttention module
multi_head_attn = MultiHeadAttention(d=16, num_heads=4)
# Inputs to the model
x1 = torch.randn(2, 10, 16)
x2 = torch.randn(2, 10, 16)
x3 = torch.randn(2, 10, 16)
