
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0)
    def forward(self, x1):
        a1 = torch.nn.functional.dropout(x1, p=0.3, training=False)
        a2 = torch.rand_like(a1)
        return self.dropout(a1) * a2
# Inputs to the model
x1 = torch.randn([10])
