
def scaled_matmul_attention(query, key, value, inv_scale_factor, dropout_p):
    qk = torch.matmul(query, key.transpose(-2, -1)) 
    return torch.nn.functional.dropout(torch.nn.functional.softmax(qk.div(inv_scale_factor), dim=-1).matmul(value), p=dropout_p)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query_projection = torch.nn.Linear(64, 32)
        self.key_projection = torch.nn.Linear(64, 32)
        self.value_projection = torch.nn.Linear(64, 32)
        self.inv_scale_factor = torch.nn.Parameter(torch.Tensor([1.0]))
        self.dropout_p = 0.5
 
    def forward(self, x1, x2):
        q = self.query_projection(x1)
        k = self.key_projection(x2)
        v = self.value_projection(x2)
        return scaled_matmul_attention(q, k, v, self.inv_scale_factor, self.dropout_p)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32, 64)
x2 = torch.randn(1, 128, 64)
