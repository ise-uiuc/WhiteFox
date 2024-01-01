
class Model(torch.nn.Module):
    def forward(self, query, key, value, attn_mask):
        return torch.dropout(query, 0.1, True)
# Inputs to the model
query = torch.randn(3, 4, 5)
key = torch.randn(3, 4, 5)
value = torch.randn(3, 4, 5)
attn_mask = torch.randn(1, 1, 4, 4)
