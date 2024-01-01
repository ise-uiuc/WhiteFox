
class Model(torch.nn.Module):
    def forward(self, x):
        f1 = torch.nn.functional.dropout(x)
        f2 = torch.nn.functional.dropout(x)
        res1 = torch.max(f1, f2)
        f3 = torch.nn.functional.dropout(x)
        f4 = torch.nn.functional.dropout(x)
        return torch.pow(res1, f3) + 0.5 * torch.abs(f4)
# Input to the model
x1 = torch.randn(1, 2, 2)
