
class Model(torch.nn.Module):
    def __init__(self, hidden_size, query_size, key_size, num_heads, dropout):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.all_head_size = hidden_size * num_heads
        self.query_map = torch.nn.Linear(query_size, self.all_head_size, bias=False)
        self.key_map = torch.nn.Linear(key_size, self.all_head_size, bias=False)
        self.value_map = torch.nn.Linear(key_size, self.all_head_size, bias=False)
        self.output_map = torch.nn.Linear(self.all_head_size, key_size)
        self.softmax_func = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, query, key, value, dropout_p):
        # shape: batch_size, query_size -> batch_size, hidden_size * num_heads
        q = self.query_map(query)
        # shape: batch_size, key_size -> batch_size, hidden_size * num_heads
        k = self.key_map(key)
        # shape: batch_size, key_size -> batch_size, hidden_size * num_heads
        v = self.value_map(value)

        # shape: (batch_size, num_heads, seq_length, hidden_size)
        q = q.view(q.size(0), self.hidden_size, self.num_heads, -1).transpose(1,2)
        k = k.view(k.size(0), self.hidden_size, self.num_heads, -1).transpose(1,2)
        v = v.view(v.size(0), self.hidden_size, self.num_heads, -1).transpose(1,2)

        # shape: (batch_size, num_heads, seq_length, key_size)
        attn = torch.matmul(q, k.transpose(-2, -1))

        # shape: (batch_size, num_heads, seq_length, key_size)
        scale_factor = torch.pow(torch.tensor(float(self.key_size))/(self.key_size**0.5), 0.5)
        scaled_attn = attn * scale_factor

        # shape: (batch_size, num_heads, seq_length, key_size)
        softmax_attn = self.softmax_func(scaled_attn)

        # shape: (batch_size, num_heads, seq_length, key_size)
        drop_attn = self.dropout(softmax_attn)
        # shape: (batch_size, seq_length, num_heads, key_size)
        drop_attn = drop_attn.transpose(1,2).contiguous()
        # shape: (seq_length, num_heads, batch_size, key_size)
        wq = drop_attn.view(drop_attn.size(0), drop_attn.size(1), -1, drop_attn.size(3))

        # shape: (batch_size, num_heads, seq_length, hidden_size)
        attn = torch.matmul(wq, v)

        # shape: (batch_size, seq_length, num_heads, key_size)
        attn = attn.contiguous().view(attn.size(0), -1, self.all_head_size)

        # shape: (batch_size, seq_length, key_size)
        output = self.output_map(attn)
        return output

# Initializing the model
query_size = 5
key_size = 6
value_size = 6
hidden_size = 4
num_heads = 3
dropout = 0.5
m = Model(hidden_size, query_size, key_size, num_heads, dropout)

# Inputs to the model
query = torch.randn(1, query_size)
key = torch.randn(2, key_size)
value = torch.randn(2, value_size)
dropout_p = 0.2
