
class Model(torch.nn.Module):
    def __init__(self, d_model, nhead, num_queries, dropout_p=0.0):
        super().__init__()
        self.scale_factor = math.sqrt(1.0 / query_num)
        self.query = torch.nn.Linear(8, d_model)
        self.key = torch.nn.Linear(8, d_model)
        self.value = torch.nn.Linear(8, d_model)
 
    def forward(self, query_vector, key_vector, value_vector):
        q = self.query(query)
        k = self.key(key_vector)
        v = self.value(value_vector)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = kq.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model(self, nhead, 32)

# Inputs to the model
query_vector = torch.randn(64, 16)
key_vector = torch.randn(64, 16)
value_vector = torch.randn(64, 16)
