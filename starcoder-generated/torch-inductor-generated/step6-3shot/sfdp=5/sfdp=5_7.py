
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dot_product = torch.nn.CosineSimilarity(dim=1)
 
    def forward(self, x1, x2):
        qk = self.dot_product(x1, x2)
        qk = qk + 1e10 * (-10000 < qk).float()
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.1, True)
        output = attn_weight @ x2
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16, 64)
x2 = torch.randn(1, 16, 64)
