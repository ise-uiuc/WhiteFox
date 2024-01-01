
class MultiHeadModel(torch.nn.Module):
    def __init__(self, batch_size, head_num, hidden_dim, dropout_p):
        super().__init__()
        self.wq = torch.nn.Linear(hidden_dim, hidden_dim)
        self.wk = torch.nn.Linear(hidden_dim, hidden_dim)
        self.wv = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        q = self.wq(query)
        k = self.wk(key)
        v = self.wv(value)

        q = q.reshape(-1, batch_size, head_num, hidden_dim // head_num)
        k = k.reshape(-1, batch_size, head_num, hidden_dim // head_num)
        v = v.reshape(-1, batch_size, head_num, hidden_dim // head_num)

        q = q.transpose(2, 1)
        k = k.transpose(2, 1)
        v = v.transpose(2, 1)

        qk = torch.matmul(q, k.transpose(2, 3))

        scaled_qk = qk.div(inv_scale_factor)

        softmax_qk = scaled_qk.softmax(dim=-1)

        dropout_qk = torch.nn.functional.dropout(torch.Tensor, p=dropout_p, training=self.training, inplace=self.training)
        

        output = dropout_qk.matmul(v)
        return output

# Inputs to the model
hidden_dim = 1024 # Number of dimensions of the hidden space
head_num = 5 # Number of heads
batch_size = 3 # Number of batches
dropout_p = 0.1 # Probability of dropout
inv_scale_factor = 1 / math.sqrt(hidden_dim // head_num) # Inverse scale factor used to scale the dot product of the query and key
query = torch.randn(batch_size * head_num, 1, hidden_dim) # Query for computing multi-head attention
key = torch.randn(batch_size * head_num, 20, hidden_dim) # Key for computing multi-head attention
value = torch.randn(batch_size * head_num, 20, hidden_dim) # Value for computing multi-head attention
