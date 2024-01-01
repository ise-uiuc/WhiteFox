
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, inv_scale_factor, dropout_p=0.8):
        v1 = torch.matmul(query, key.transpose(-2, -1))
        v2 = inv_scale_factor / v1
        v3 = torch.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=dropout_p)
        v5 = torch.matmul(v4, value)
        return v5
query = torch.randn(64, 32, 100)
key = torch.randn(64, 32, 200)
value = torch.randn(64, 32, 200)
inv_scale_factor = 100
