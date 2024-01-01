
class Model(torch.nn.Module):
    def forward(self, x):
        b = torch.nn.functional.dropout(x, p=0.2)
        a = torch.rand_like(x)
        return a
# Inputs to the model
X = torch.rand([2, 2])
