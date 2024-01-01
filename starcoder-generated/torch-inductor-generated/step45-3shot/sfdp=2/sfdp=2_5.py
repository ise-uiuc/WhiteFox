
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout0 = torch.nn.Dropout(0.1)
 
    def forward(self, q, k, v, inv_scale_factor):
        q_k = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = q_k.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout0(softmax_qk)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(16, 1, 64)
k = torch.randn(16, 1, 64)
v = torch.randn(16, 1, 64)
inv_scale_factor = torch.randn(16, 16, 1)
