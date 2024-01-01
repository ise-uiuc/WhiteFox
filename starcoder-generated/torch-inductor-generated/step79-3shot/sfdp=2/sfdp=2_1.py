
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        inv_scale_factor = torch.pow(x1.shape[-1], -0.25).to(x1.dtype)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(x2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 80)
x2 = torch.randn(1, 80, 64)
