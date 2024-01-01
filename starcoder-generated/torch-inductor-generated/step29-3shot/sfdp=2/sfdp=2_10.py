
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = math.sqrt((k.shape[-1])**(-0.25))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = torch.nn.functional.softmax(scaled_qk, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 768, 64, 64)
k = torch.randn(1, 768, 64, 64)
v = torch.randn(1, 768, 64, 64)
