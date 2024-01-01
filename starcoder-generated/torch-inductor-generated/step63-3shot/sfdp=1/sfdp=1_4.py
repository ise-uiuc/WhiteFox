
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(16, 384, 384))
        self.key = torch.nn.Parameter(torch.randn(16, 384, 4096))
        self.value = torch.nn.Parameter(torch.randn(16, 4096, 4096))
 
    def forward(self, q, k, v, alpha):
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = (q.size(-1) ** -0.5) * (k.size(-1) ** -0.5)
        scale_factor = alpha * inv_scale_factor
        scaled_qk = qk * scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_p = 0.1
        dropout_qk = F.dropout(softmax_qk, dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 768, 80)
k = torch.randn(1, 768, 128)
v = torch.randn(1, 128, 2048)
alpha = 0.2
