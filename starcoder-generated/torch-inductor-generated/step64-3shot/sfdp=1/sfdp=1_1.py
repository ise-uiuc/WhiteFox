
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads=8, scale_factor=1):
        super().__init__()
        self.query = torch.nn.Linear(dim, dim)
        self.key = torch.nn.Linear(dim, dim)
        self.value = torch.nn.Linear(dim, dim)
        self.inv_scale_factor = scale_factor ** -0.5
 
    def forward(self, query, key, value, dropout_prob):
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
        # print(f"key={k.shape}, key_tranposed={k.transpose(-2, -1).shape}, query={q.shape}, inv_scale_factor={self.inv_scale_factor}")
        qk = torch.matmul(q, k.transpose(-2, -1))
        # print(f"qk={qk}")
        scaled_qk = qk * self.inv_scale_factor
        # print(f"scale_qk={scaled_qk})
        softmax_qk = scaled_qk.softmax(dim=-1)
        # print(f"softmax_qk={softmax_qk}")
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_prob)
        # print(f"dropout_qk={dropout_qk}")
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model(dim=1024, num_heads=8)

# Inputs to the model
query = torch.randn(1, 1, 1024)
key = torch.randn(1, 4, 1024)
value = torch.randn(1, 4, 1024)
dropout_prob = 0.1
