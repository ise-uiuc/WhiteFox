
class Model(torch.nn.Module):
    def __init__(self, input_embedding_size, num_heads, dropout_p):
        super().__init__()
        self.input_embedding_size = input_embedding_size
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.q = torch.nn.Linear(input_embedding_size, num_heads)
        self.k = torch.nn.Linear(input_embedding_size, num_heads)
        self.v = torch.nn.Linear(input_embedding_size, num_heads)
        self.inv_scale_factor = torch.nn.Parameter(torch.zeros((1, self.num_heads, 1, 1), dtype=torch.float32))
 
    def forward(self, query, key, value, mask):
        q = self.q(query)
        k = self.k(key)
        v = self.v(value)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        if mask is not None:
            scaled_qk = mask.fill_attention_mask(scaled_qk, -1e9)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, seq_len, input_embedding_size)
key = torch.randn(1, seq_len, input_embedding_size)
value = torch.randn(1, seq_len, input_embedding_size)
mask = torch.tril(torch.ones((batch_size, 1, seq_len, seq_len), dtype=torch.uint8))
