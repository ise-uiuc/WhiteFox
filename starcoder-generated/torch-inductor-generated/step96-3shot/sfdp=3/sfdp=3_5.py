
class SelfAttention(torch.nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.query_projection = torch.nn.Linear(d_model, d_model)
        self.key_projection = torch.nn.Linear(d_model, d_model)
        self.value_projection = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, query, key, value):
        queries = self.query_projection(query)
        keys = self.key_projection(key)
        values = self.value_projection(query)
        QK = torch.matmul(queries, keys)
        scale_factor = (self.d_model / self.n_head) ** -0.5
        attention = QK.mul(scale_factor).softmax(dim=-1)
        attention = self.dropout(attention)
        attention = torch.matmul(attention, values)
        return attention

# Initializing the model
d_model = 512
n_head = 8
d_head = d_model // n_head
m = SelfAttention(d_model, n_head)

# Inputs to the model
query = torch.randn(1, 17, d_model)
key = torch.randn(1, 19, d_model) # The number of rows in key is not the same as that in query
value = torch.randn(1, 19, d_model)
