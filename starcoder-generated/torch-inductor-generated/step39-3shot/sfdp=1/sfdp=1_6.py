
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def make_query_key_value(self, m, q, k, v):
        mk, qk, vk = m.split(m.shape[0] // 3, dim=0)
        return qk, mk, vk
 
    def forward(self, q, k, v):
        qk, mk, vk = self.make_query_key_value(q, q, k, v)
        scale_factor = (mk.shape[0] * mk.shape[-1])**(-0.25)
        inv_scale_factor = 1 / scale_factor
        qk = qk.matmul(k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.2)
        return dropout_qk.matmul(v)

# Initializing the model
m = Model()

# Inputs to the model
m = Model()
q = torch.randn(14, 20, 10)
k = torch.randn(20, 3, 80)
v = torch.randn(20, 80)
