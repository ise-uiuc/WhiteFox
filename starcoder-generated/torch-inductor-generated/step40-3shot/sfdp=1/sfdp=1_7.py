
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v, scale_factor, d_k, dropout_p=0.1):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output
        
# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 2, 2, 2)
k = torch.randn(1, 2, 2, 2)
v = torch.randn(1, 2, 2, 2)
