
class Model(torch.nn.Module):
    def forward(self, q, k, v, inv_scale_factor, dropout_p, mask=None):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        if mask is not None:
            dropout_qk.masked_fill_(mask, 0)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 100, 128)
k = torch.randn(1, 100, 128)
v = torch.randn(1, 100, 256)
inv_scale_factor = torch.randn(1)
dropout_p = torch.tensor(0.0)
