
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, attn_mask):
        result = torch.nn.functional.normalize(q, p=2, dim=1) @ torch.transpose(torch.nn.functional.normalize(k, p=2, dim=1), -2, -1)
        return result@v + attn_mask

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 5, 15)
k = torch.randn(1, 5, 20)
v = torch.randn(1, 20, 15)
attn_mask = torch.randn(15, 20).to(torch.bool)
