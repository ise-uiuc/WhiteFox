
class SelfAttention(torch.nn.Module):
    def __init__(self, hidden_size, dropout_p):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.linear_q = torch.nn.Linear(...)
        self.linear_k = torch.nn.Linear(...)
        self.linear_v = torch.nn.Linear(...)
 
    def forward(self, input):
        bs, seq_len, hidden_size = input.shape
        query = self.linear_q(input).reshape(bs, seq_len, self.num_heads, -1)
        key = self.linear_k(input).reshape(bs, seq_len, self.num_heads, -1)
        value = self.linear_v(input).reshape(bs, seq_len, self.num_heads, -1)
        scaled_qk = torch.softmax(torch.matmul(query, key) / math.sqrt(self.hidden_size / self.num_heads), dim=-1)
        dropout_qk = torch.nn.functional.dropout(scaled_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value.transpose(1, 2))
        return output.reshape(bs, seq_len, -1)

# Initializing the attention module
m = SelfAttention(16, 0.0)

# Inputs to the attention module
x = torch.randn(1, 5, 16)
