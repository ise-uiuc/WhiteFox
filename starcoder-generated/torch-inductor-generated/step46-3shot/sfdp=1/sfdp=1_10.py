
class Model(torch.nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.query_projection = torch.nn.Linear(3, self.num_heads)
        self.key_projection = torch.nn.Linear(3, self.num_heads)
        self.value_projection = torch.nn.Linear(3, self.num_heads)
 
    def forward(self, query, key, value, dropout_p):
        q = self.query_projection(query)
        k = self.key_projection(key)
        v = self.value_projection(value)
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = torch.float32(1. / np.sqrt(float(q.shape[-1])))
        scaled_qk = qk.div(inv_scale_factor)
        dropout_qk = torch.nn.functional.dropout(scaled_qk, p=dropout_p)
        return dropout_qk.matmul(v)

# Initializing the model
m = Model(num_heads=5)

# Inputs to the model
query = torch.randn(4, 3, 60)
key = torch.randn(4, 120, 3)
value = torch.randn(4, 120, 3)
dropout_prob = 0.2
