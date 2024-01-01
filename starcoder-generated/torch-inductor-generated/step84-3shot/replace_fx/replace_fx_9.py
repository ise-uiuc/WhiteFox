
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = lowmem_dropout(x, 0.6, True)
        x = F.dropout(x, p=0.4)
        return x
# Inputs to the model
x = torch.randn(1, 1, 2)
