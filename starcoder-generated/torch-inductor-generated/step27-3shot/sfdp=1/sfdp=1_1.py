
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(5, 8, bias=False)
        self.key = torch.nn.Linear(5, 8, bias=False)
        self.value = torch.nn.Linear(5, 8, bias=False)
        self.inv_scale_factor = torch.nn.Parameter(torch.finfo(torch.float32).tiny)
 
    def forward(self, query, key, value, dropout_p):
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 3, 5)
key = torch.randn(1, 6, 5)
value = torch.randn(1, 6, 5)
dropout_p = 0.1
