
class Model(torch.nn.Module):
    def __init__(self, query_seq_len, key_seq_len, hidden_size):
        super().__init__()
        self.w1 = torch.nn.Linear(query_seq_len, hidden_size, bias=False)
        self.w2 = torch.nn.Linear(key_seq_len, hidden_size, bias=False)
        self.dropout = torch.nn.Dropout(p=0.5)
 
    def forward(self, q, k):
        q = self.w1(q)
        k = self.w2(k)
        q_k = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = math.sqrt(math.sqrt(q_k.size()[-1]))
        scaled_q_k = q_k.div(inv_scale_factor)
        softmax_q_k = scaled_q_k.softmax(dim=-1)
        dropout_q_k = self.dropout(softmax_q_k)
        output = dropout_q_k.matmul(v)
        return output

# Initializing the model
query_seq_len = 128
key_seq_len = 1024
hidden_size = 256
m = Model(query_seq_len, key_seq_len, hidden_size)

# Inputs to the model
x1 = torch.randn(1, query_seq_len, hidden_size)
x2 = torch.randn(1, key_seq_len, hidden_size)
