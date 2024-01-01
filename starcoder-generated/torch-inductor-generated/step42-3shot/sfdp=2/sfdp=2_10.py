
class Model(torch.nn.Module):
    def forward(self, queries, keys, values, dropout=0.5, scale_factor=None, mask=None):
        qk = torch.matmul(queries, keys.transpose(-2, -1))
        if scale_factor is not None:
            inv_scale_factor = 1 / scale_factor
            scaled_qk = qk.div(inv_scale_factor)
        else:
            scaled_qk = qk
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout)
        output = dropout_qk.matmul(values)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 128, 256)
x2 = torch.randn(1, 3, 256, 128)
x3 = torch.randn(1, 3, 256, 512)
