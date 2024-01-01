
class Model(torch.nn.Module):
    def __init__(self, attention_n_head=5, attention_head_size=16, attention_dropout_rate=0.1):
        super().__init__()
        self.attention_n_head = attention_n_head
        self.attention_head_size = attention_head_size
        self.attention_dropout_rate = attention_dropout_rate
        self.dense = torch.nn.Linear(256, attention_n_head * self.attention_head_size)
        self.query_layer = torch.nn.Linear(256, attention_n_head * self.attention_head_size)
        self.key_layer = torch.nn.Linear(256, attention_n_head * self.attention_head_size)
        self.value_layer = torch.nn.Linear(256, attention_n_head * self.attention_head_size)
        self.dropout = torch.nn.Dropout(attention_dropout_rate)
        self.out = torch.nn.Linear(attention_n_head * self.attention_head_size, 256)
 
    def forward(self, query, key, value, mask):
        batch_size = value.shape[0]
        hidden_size = value.shape[-1]
        scaled_dot = torch.matmul(query, key.transpose(-2, -1)) \
             / math.pow(self.attention_head_size, 0.5)
        output = torch.softmax(scaled_dot, dim=-1) + mask[:, None, None, :]
        output = self.dropout(output)
        output = torch.matmul(output, value)
        concatenated = output.view(batch_size, -1, self.n_head * self.attention_head_size)
        return self.out(concatenated), output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 8, 256)
key = torch.randn(1, 8, 256)
value = torch.randn(1, 8, 256)
mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1]])
__output__, __attention__ = m(query, key, value, mask)
