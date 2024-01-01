
class Model(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super(__class__, self).__init__()
        self.hidden_size = hidden_size
 
        self.query = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value):
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output
```
# Inputs to the model
x1 = torch.randn(1, hidden_size)
x2 = torch.randn(1, hidden_size)
x3 = torch.randn(1, hidden_size)
