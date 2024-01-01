
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v, scale_factor, dropout_p, dropout_apply_pos):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)-
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        if dropout_apply_pos == 'head':
            output[:,0,:] = output[:,1,:]
            output[:,1,:] = output[:,2,:]
            output[:,2,:] = output[:,3,:]
        elif dropout_apply_pos == 'all':
            output[:,:0,:] = output[:,1,:]
            output[:,1:,:] = output[:,2,:]
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(32, 8, 64)
k = torch.randn(32, 8, 64)
v = torch.randn(32, 8, 64)
scale_factor = 1.0 / (4 * 64 ** 0.5)
dropout_p = 0.1
dropout_apply_pos = 'none'
