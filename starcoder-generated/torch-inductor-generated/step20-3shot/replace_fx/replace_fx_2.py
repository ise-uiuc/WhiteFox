
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_layer = torch.nn.Dropout(inplace=1)
    def forward(self, x1):
        x2 = torch.rand_like(x1).to(dtype=torch.float16)
        x3 = self.dropout_layer(x2)
        return x3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
