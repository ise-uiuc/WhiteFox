
class Model(torch.nn.Module):
    def __init__(self, d_model, num_heads, dropout_p):
        super().__init__()
        self.query_layer = torch.nn.Linear(d_model, d_model)
        self.key_layer = torch.nn.Linear(d_model, d_model)
        self.value_layer = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, q, k, v):
        q = self.query_layer(q)
        k = self.key_layer(k)
        v = self.value_layer(v)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(math.sqrt(torch.tensor(k.size(-1))))
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model(d_model=128, num_heads=8, dropout_p=0.3)

# Inputs to the model
q = torch.randn(1, 16, 128)
k = torch.randn(1, 32, 128)
v = torch.randn(1, 32, 128)
