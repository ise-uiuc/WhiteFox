
class model(torch.nn.Module):
    def forward(self, x):
        a = x.mean()
        return (F.dropout(a, 1e-2) * a).pow(0)
# Inputs to the model
x1 = torch.randn(1, 2, 2) 
