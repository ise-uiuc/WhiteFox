
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.permute(0, 1, 3, 2))
        inv_scale_factor = math.sqrt(x1.shape[-1])
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, 0.1)
        v1 = dropout_qk.matmul(x2)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 15, 20, 20)
x2 = torch.randn(1, 20, 1, 100)
