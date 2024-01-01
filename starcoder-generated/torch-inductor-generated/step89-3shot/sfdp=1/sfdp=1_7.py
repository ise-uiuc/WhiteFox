
class Model(torch.nn.Module):
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = self.scale_factor.pow(-1)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(2, 3, 4)
k = torch.randn(2, 4, 6)
v = torch.randn(2, 4, 6)
dropout_p = 0.3
