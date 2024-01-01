
class Model(torch.nn.Module):
    def forward(self, x):
        v1 = torch.rand_like(x, dtype=torch.float)
        v2 = torch.nn.functional.dropout(x, p=0.1)
        z = torch.nn.functional.gelu(x)
        u = torch.rand()
        r = (v1+v2)*z
        return r
# Inputs to the model
x = torch.randn(2, 20, 20)
