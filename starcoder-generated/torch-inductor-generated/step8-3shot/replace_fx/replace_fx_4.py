
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        t1 = torch.nn.functional.dropout(torch.randn(2, 2), p=0.5, training=True)
        t2 = torch.rand_like(t1)
        t3 = torch.argmax(t2, axis=1)
        return t1, t3, t2
# Inputs to model
