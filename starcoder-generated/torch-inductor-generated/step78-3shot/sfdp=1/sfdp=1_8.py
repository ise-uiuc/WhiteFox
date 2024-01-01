
class Model(torch.nn.Module):
    def forward(self, q1, k1):
        v1 = torch.matmul(q1, k1.transpose(-2, -1))
        inv_scale_factor = 0.7
        v2 = v1 * inv_scale_factor
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        dropout_p = 0.0
        v4 = torch.nn.functional.dropout(v3, p=dropout_p)
        v5 = torch.matmul(v4, v3)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(1, 3, 16, 16)
k1 = torch.randn(1, 3, 32, 32)
