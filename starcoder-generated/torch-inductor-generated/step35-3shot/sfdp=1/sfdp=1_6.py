
class Model(torch.nn.Module):
    def __init__(self, query_size, key_size, value_size, dropout_p, inv_scale_factor):
        super().__init__()
        self.query_projection = torch.nn.Linear(q, k)
        self.key_projection = torch.nn.Linear(k, k)
        self.value_projection = torch.nn.Linear(v, v)
        self.inv_scale_factor = inv_scale_factor
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value):
        q = self.query_projection.forward(query)
        k = self.key_projection.forward(key)
        v = self.value_projection.forward(value)
        qk = torch.matmul(q, k.t())
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
query_size = 100
key_size = 100
value_size = 200
dropout_p = 0.5
inv_scale_factor = torch.sqrt(torch.FloatTensor([key_size]))
m = Model(query_size, key_size, value_size, dropout_p, inv_scale_factor)

# Inputs to the model
query = torch.randn(5, query_size)
key = torch.randn(10, key_size)
value = torch.randn(10, value_size)
