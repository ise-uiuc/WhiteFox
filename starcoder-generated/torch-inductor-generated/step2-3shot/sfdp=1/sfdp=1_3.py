
class Model(torch.nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.scale_factor = 1 / math.sqrt(model_dim)
 
    def forward(self, x1, x2):
        v1 = F.nll_loss(x1, x2)
        v2 = v1 * self.scale_factor
        v3 = v2.softmax(dim=1)
        v4 = torch.nn.functional.dropout(v3, p=0.5)
        v5 = F.nll_loss(v4, torch.transpose(v4, 0, 1))
        return v5

# Initializing the model
m = Model(16)

# Inputs to the model
x1 = torch.randn(16, 16, 5, 5)
x2 = torch.zeros([16], dtype=torch.int64)
