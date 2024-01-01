
class Model(torch.nn.Module):
    def __init__(self, dim_k, dropout_p=0):
        super().__init__()
        self.dropout_p = dropout_p
        self.softmax = torch.nn.Softmax(-1)
        self.dropout = torch.nn.Dropout(p=dropout_p)
 
    def forward(self, q, k, v, inv_scale_factor):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
dim_k = 16
dropout_p = 0.2
m = Model(dim_k, dropout_p)

# Inputs to the model
q = torch.randn(1, 8, dim_k)
k = torch.randn(1, 7, dim_k)
v = torch.randn(1, 7, dim_k)
inv_scale_factor = 1. / math.sqrt(dim_k)
