
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = 1 / np.sqrt(q.shape[-1])
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.rand(1, 1, 128)
k = torch.rand(1, 1, 128)
v = torch.rand(1, 1, 128)
