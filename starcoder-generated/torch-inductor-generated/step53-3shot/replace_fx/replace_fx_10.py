
class model(torch.nn.Module):
    def forward(self, x):
        x = F.dropout(x, p=0.5)
        x = F.dropout(x, p=0)
        return x
# Inputs to the model
x1 = [torch.randn(1, 2, 2), torch.randn(1, 2, 2)]
