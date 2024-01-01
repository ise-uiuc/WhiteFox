
class Model(torch.nn.Module):
    def __init__(self, n_head=8):
        super().__init__()
        self.n_head = n_head
        
    def forward(self, q, k, v, inv_scale_factor=1, dropout_p=0.2):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor**0.5)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model(n_head=8)

# Random input tensor
input_tensor = torch.randn(1, 64, 200)

# Inputs to the model
q = torch.randn(1, 8, 64)
k = torch.randn(1, 8, 200)
v = torch.randn(1, 8, 200)
