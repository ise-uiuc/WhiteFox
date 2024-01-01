
class Model(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(d_model, d_model))
        self.key = torch.nn.Parameter(torch.randn(d_model, d_model))
        self.value = torch.nn.Parameter(torch.randn(d_model, d_model))

    def forward(self, x1, x2):
        qk = torch.matmul(x1, self.key.transpose(-2, -1))
        scale_factor = (np.power(self.query.shape[-1], -0.5)).item()
        v3 = qk.mul(scale_factor)
        v4 = v3.softmax(dim=-1)
        v5 = torch.nn.functional.dropout(v3, p=0.5)
        v6 = v4.matmul(v5)
        return v6

# Initializing the model
m = Model(256, 4)

# Inputs to the model
x1 = torch.randn(1, 256, 256)
x2 = torch.randn(1, 256, 256)
