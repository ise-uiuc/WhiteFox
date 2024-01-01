
class Model(torch.nn.Module):
    def forward(self, x):
        f1 = torch.nn.functional.dropout(x)
        f2 = torch.nn.functional.dropout(x)
        res1 = torch.sub(f2, f1)
        f3 = torch.nn.functional.dropout(x)
        f4 = torch.nn.functional.dropout(x)
        return torch.atan(0.5 * res1) + torch.exp(f4)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
