
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q1, k2):
        q2 = torch.nn.functional.normalize(q1)
        k3 = torch.nn.functional.normalize(k2)
        q3 = q2 * q2
        q4 = q3.sum(dim=2)
        k4 = k3.sum(dim=2)
        qq = torch.matmul(q4, k4.transpose(-2, -1))
        inv_scale_factor = (q2.shape[-1] * q2.shape[-2]).to('cpu').numpy() ** -0.5
        scaled_qk = qq.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        v5 = torch.matmul(dropout_qk, k3)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(1, 768, 196)
k2 = torch.randn(1, 768, 100)
