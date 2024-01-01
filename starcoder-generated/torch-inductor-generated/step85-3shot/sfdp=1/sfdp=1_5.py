
class Model(torch.nn.Module):
    def __init__(self, d_model, n_heads, dropout_p, inv_scale_factor):
        super().__init__()
        self.query = torch.nn.Conv2d(3, n_heads, 1, stride=1, padding=1)
        self.key = torch.nn.Conv2d(3, n_heads, 1, stride=1, padding=1)
        self.value = torch.nn.Conv2d(3, n_heads, 1, stride=1, padding=1)
        self.dropout_p = dropout_p
        self.inv_scale_factor = inv_scale_factor

    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model(3, 8, 0.0, 2)

# Inputs to the model
q = torch.randn(1, 3, 64, 64)
k = torch.randn(1, 3, 64, 64)
__v__ = torch.randn(1, 3, 64, 64)
