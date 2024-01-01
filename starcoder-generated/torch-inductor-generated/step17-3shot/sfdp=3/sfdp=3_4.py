
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qk_mat = torch.nn.Linear(32, 32)
 
    def forward(self, q, k, v, mask):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scale_factor = (k.size(-1) ** -0.5)
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.7)
        qk_dropout = dropout_qk.matmul(v)
        return qk_dropout

# Initializing the model
m = Model()

# Inputs to the model
_q = torch.randn(16, 32)
_k = torch.randn(16, 32)
_v = torch.randn(16, 32)
_mask = torch.ones(_q.size(0), _k.size(0))
