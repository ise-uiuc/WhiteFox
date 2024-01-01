
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
self.query_key_value = torch.nn.Linear(60, 60)
self.query_dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, x1):
        x2 = self.query_key_value(x1)
        qk = x1.view(-1, 1, 5, 6)
        kv = x2.view(-1, 5, 1, 6)
        qk = torch.matmul(qk, kv)
        qk = qk.div(inv_scale_factor)
        softmax_qk = qk.softmax(dim=-1)
        dropout_qk = self.query_dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(150, 60)
