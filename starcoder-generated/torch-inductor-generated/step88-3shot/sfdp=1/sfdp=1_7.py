
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, q, k, v, inv_scale_factor, dropout_p):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Initializing input tensors and parameters
q = torch.randn(2, 7, 8)
k = torch.randn(2, 8, 5)
v = torch.randn(2, 8, 5)
inv_scale_factor = torch.randn(8)
dropout_p = torch.rand(())
