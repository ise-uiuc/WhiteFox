
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout2d(0.5)
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = q.shape[-1] ** -0.5
        scaled_qk = qk.mul(inv_scale_factor)
        softmax_qk = torch.nn.functional.softmax(scaled_qk, dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        return dropout_qk.matmul(v)

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 8, 30, 64)
k = torch.randn(1, 8, 42, 64)
v = torch.randn(1, 8, 42, 64)
