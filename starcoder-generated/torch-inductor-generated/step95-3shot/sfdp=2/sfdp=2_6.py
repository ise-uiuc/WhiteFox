
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q1, k2, v3):
        qk = torch.matmul(q1, k2)
        inv_scale_factor = (self.head_dim ** -0.5)
        scaled_qk = qk * inv_scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropdout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropdout_qk * v3
        return output

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(1, 4, 256)
k2 = torch.randn(1, 4, 256)
v3 = torch.randn(1, 4, 64, 256)
