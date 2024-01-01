
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, q, k, v, inv_scale_factor):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model() 

# Inputs to the model
q = torch.randn(1, 2, 512)
k = torch.randn(1, 4, 512)
v = torch.randn(1, 4, 512)
inv_scale_factor = torch.randn(1, 1)
