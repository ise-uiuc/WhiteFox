
class Model(torch.nn.Module):
    def forward(self, x):
        x1 = torch.nn.functional.dropout(x, p=0.37)
        x2 = torch.nn.functional.dropout(x, p=0.28)
        x3 = torch.nn.functional.dropout(x, p=0.73)
        x4 = torch.nn.functional.dropout(x, p=0.52)
        return torch.mm(x1, x2) - x3 - x4 + 2.0
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
