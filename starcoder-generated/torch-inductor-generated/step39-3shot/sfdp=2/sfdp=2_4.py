
class Model(torch.nn.Module):
    def forward(self, q, k, v):
        scaled_qk = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.shape[-1])
        softmax_qk = scaled_qk.softmax(dim=-1) * dropout_p
        output = softmax_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 8, 32, 32)
k = torch.randn(1, 8, 32, 32)
v = torch.randn(1, 8, 32, 32)
