
class Model(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
 
    def forward(self, q, k, v):
        batch, heads, length = self.shape
        q = torch.nn.functional.elu(q)
        k = torch.nn.functional.elu(k)
        v = torch.nn.functional.elu(v)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scale_factor = np.reciprocal(np.sqrt(q.size(-1)))
        inv_scale_factor = 1 / scale_factor
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = torch.nn.functional.gumbel_softmax(scaled_qk, hard=True)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
dummy_shape = (1, 1, 9)
m = Model(dummy_shape)

# Inputs to the model
q = torch.randn(9, 1)
k = torch.randn(9, 1)
v = torch.randn(9, 1)
