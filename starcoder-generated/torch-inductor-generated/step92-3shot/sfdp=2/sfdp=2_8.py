
class Model(torch.nn.Module):
    def forward(self, q, k, v, scale_factor, dropout_p):
        inv_scale_factor = 1 / scale_factor
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(3, 10, 8)
k = torch.randn(3, 8, 6)
v = torch.randn(3, 8, 10)
