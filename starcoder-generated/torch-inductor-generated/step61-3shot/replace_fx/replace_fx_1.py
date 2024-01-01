
class model(torch.nn.Module):
    def forward(self, x):
        a1 = torch.nn.functional.dropout(x, p=0.6)
        if a1 and torch.rand(1) > 0.5:
            return True
        else:
           return False
# Inputs to the model
x = torch.rand(2, 2)
