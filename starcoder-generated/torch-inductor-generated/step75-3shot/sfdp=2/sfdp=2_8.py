
class Model(torch.nn.Module):
    def forward(self, q, k, v, isf):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(isf)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.0)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(3, 4, 8)
k = torch.randn(3, 6, 8)
v = torch.randn(3, 6, 8)
isf = torch.full((4, 8), 1.0 / (1.0 + 8), dtype=torch.float32)
