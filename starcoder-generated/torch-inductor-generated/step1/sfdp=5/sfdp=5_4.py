
class Model(torch.nn.Module):
    def forward(self, x):
        v1 = x @ x.transpose(-2, -1)
        v2 = v1 / np.sqrt(x.shape[-1])
        v3 = v2 + attn_mask
        v4 = torch.softmax(v3, dim=-1)
        v5 = torch.nn.functional.dropout(v4, p=dropout_p, train=True)
        v6 = v5 @ value
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 2, 64)
