
class Model(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
 
    def forward(self, q, k, v, scale_factor=None, dropout_p=0.):
        q_k = torch.matmul(q, k.transpose(-2, -1))
        if scale_factor is not None:
            q_k = q_k.div(scale_factor)
        softmax_qk = torch.nn.functional.softmax(q_k, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1, training=self.training)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model(2)

# Inputs to the model
q = torch.randn(1, 5, 2)
k = torch.randn(1, 6, 2)
v = torch.randn(1, 6, 2)
