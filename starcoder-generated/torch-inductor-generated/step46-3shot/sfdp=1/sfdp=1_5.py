
class Model(torch.nn.Module):
    def __init__(self, dropout_p, d_model, nhead):
        super(Model, self).__init__()
        self.dropout_p = dropout_p
        self.d_model = d_model
        self.nhead = nhead
 
        self.qk_weight = torch.nn.Parameter(torch.Tensor(d_model, nhead, d_model))
        self.qk_bias = torch.nn.Parameter(torch.Tensor(nhead, d_model))
        self.v_weight = torch.nn.Parameter(torch.Tensor(d_model, nhead, d_model))
        self.v_bias = torch.nn.Parameter(torch.Tensor(nhead, d_model))
 
        self.dropout = torch.nn.Dropout(p=dropout_p)
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        qk = qk.div(math.sqrt(self.d_model))
        qk = qk.add_(self.qk_bias[:, :, None])
 
        v = torch.matmul(value, self.v_weight)
        v = v.add_(self.v_bias)
 
        dropout_qk = self.dropout(torch.nn.functional.softmax(qk, dim=-1))
        output = dropout_qk.matmul(v)
        output = torch.transpose(output, 1, 2)
        return output

# Initializing values of required parameters
dropout_p = 0.1
d_model = 256
nhead = 16
 
# Initializing the model
m = Model(dropout_p, d_model, nhead)
 
# Inputs to the model
query = torch.randn(1, 128, d_model)
key = torch.randn(1, 256, d_model)
value = torch.randn(1, 256, d_model)
