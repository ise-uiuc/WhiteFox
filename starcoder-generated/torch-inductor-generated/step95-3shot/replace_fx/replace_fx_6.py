
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t1 = torch.nn.Dropout2d(p=0.5)
        t2 = t1(x1) # Use instance method of dropout2d to trigger a subgraph that has a pattern node that has args and is thus replaced
        t3 = torch.rand_like(x2)
        t4 = torch.nn.functional.dropout(t3, p=0.3) # Dropout has p=0.5, however rand_like will generate random values with p=0
        t5 = torch.nn.functional.dropout(t2, p=0.5) # Dropout has p=0.5, which is already in the subgraph of t1, and t3 does not affect subgraph of t1, so this will be replaced as well
        return (t4, t5)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1)
