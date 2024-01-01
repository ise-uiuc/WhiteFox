
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v, scale_factor, dropout_p):
        scale_factor = q[:, 0::q.size(1) // 4].reshape(q.shape[0], -1).contiguous()
        dropout_p = q[:, q.shape[1] // 4: q.shape[1] * 3 // 4].reshape(q.shape[0], -1).contiguous()
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor.to(k.dtype).to(qk.device))
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        dropout_qk_value = dropout_qk.matmul(v)
        return dropout_qk_value

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(2, 8, 16)
k = torch.randn(2, 8, 16)
v = torch.randn(2, 8, 32)
scale_factor = torch.randn(2)
dropout_p = torch.randn(2)
