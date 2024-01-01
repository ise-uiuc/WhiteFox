
class Model(torch.nn.Module):
    def forward(self, *x):
        q, k, v, inv_scale_factor, dropout_p = x
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        y = dropout_qk.matmul(v)
        return y

# Initializing the model
m = Model()

# Inputs to the model
shape = 1, 128, 256
inv_scale_factor = torch.randn(shape)
dropout_p = 0.95
x1 = torch.randn(shape)
x2 = torch.randn(shape)
x3 = torch.randn(shape)
