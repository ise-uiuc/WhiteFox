
class SelfAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_p):
        super().__init__()
        self.key = torch.nn.Linear(hidden_size, hidden_size)
        self.query = torch.nn.Linear(hidden_size, hidden_size)
        self.value = torch.nn.Linear(hidden_size, hidden_size)
        self.fc = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        k = self.key(x2) 
        q = self.query(x1) 
        v = self.value(x2) 
        qk = torch.matmul(q, k.transpose(-2, -1)) 
        scaled_qk = qk.div(inv_scale_factor) 
        softmax_qk = self.softmax(scaled_qk) 
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p) 
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
s = SelfAttention(hidden_size, num_attention_heads, dropout_p)

# Inputs to the model
__input1__ = torch.randn(max_seq_len, batch_size, hidden_size) 
__input2__ = torch.randn(batch_size, max_seq_len, hidden_size) 
