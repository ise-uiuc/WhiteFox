
class Model(torch.nn.Module):
    def __init__(self, query_seq_len, key_seq_len, num_heads):
        super().__init__()
        self.query_seq_len = query_seq_len
        self.key_seq_len = key_seq_len
        self.num_heads = num_heads
        self.dropout = torch.nn.Dropout(0.1)

        self.to_q = torch.nn.Linear(2, 4)
        self.to_k = torch.nn.Linear(2, 4)
        self.to_v = torch.nn.Linear(2, 4)

    def forward(self, input):
        query = self.to_q(input).view(self.num_heads, self.query_seq_len, seq_len, 1)
        key = self.to_k(input).view(-1, self.num_heads, self.key_seq_len, 1)
        value = self.to_v(input)

        qk = query.matmul(key.transpose(-1, -2)) # Compute the dot product of the query and the key
        scaled_qk = qk / math.sqrt(2) # Scale the dot product by 1/sqrt(2)
        dropout_qk = torch.nn.functional.dropout(scaled_qk, p=0.2) # Apply dropout to the scaled dot product
        output = dropout_qk.matmul(value) # Compute the dot product of the dropout output and the value
        return output

# Initializing the model
device = 'cpu'
m = Model(seq_len, seq_len, 4)
m.to(device)

# Inputs to the model
x1 = torch.randn(1, 2, 100)
