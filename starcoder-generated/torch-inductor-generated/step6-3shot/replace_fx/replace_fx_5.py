
class Model(torch.nn.Module):
    def forward(self,x):
        a = torch.randint(0, 2, (2, 2), dtype=torch.float)
        b = torch.nn.functional.dropout(a, p=0.2)
        c = torch.nn.functional.dropout(b, p=0.1, training=True)
        d = torch.nn.functional.dropout(b, p=0.25, training=False)
        return torch.nn.functional.dropout(c)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
