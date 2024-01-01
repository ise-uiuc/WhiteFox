
scale_factor = (d**-0.25)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = scale_factor 
        self.d = d
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2,-1))
        inv_scale_factor = self.scale_factor / self.d
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs for the model
input1 = torch.randn(1, seq_dim, dim)
input2 = torch.randn(1, seq_dim, dim)
input3 = torch.randn(1, seq_dim, dim)
