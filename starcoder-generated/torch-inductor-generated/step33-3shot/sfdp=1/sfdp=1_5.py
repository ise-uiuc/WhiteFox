
class Model(torch.nn.Module):
    def __init__(self, query, key, value):
        super().__init__()
        self.q_key = torch.nn.Linear(query_size, key_size, bias=False)
        self.v = torch.nn.Linear(value_size, output_size)
        self.qk_norm = 1024 ** -0.5
        self.dropout = torch.nn.Dropout(dropout_p, _mode="2d")
 
    def compute_attention(self, q, k, v):
        # Perform scaled-dot attention
        qk = self.q_key(q).transpose(-2, -1)
        qk = qk / self.qk_norm
        scaled_qk = torch.matmul(qk, k)
        softmax_qk = scaled_qk.softmax(dim=-1)
        return self.dropout(softmax_qk), torch.matmul(softmax_qk, v)
 
    def forward(self, q, k, v):
        # Do one pass of scaled-dot attention
        attention, val = self.compute_attention(q, k, v)
        output = self.v(val)
        return output

# Initializing the model
q = torch.randn(batch_size, query_size, seq_length)
k = torch.randn(batch_size, key_size, seq_length)
v = torch.randn(batch_size, value_size, seq_length)
