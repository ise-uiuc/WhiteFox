
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        t0 = torch.rand(1)
        t1 = torch.nn.functional.dropout(t0)
        t2 = torch.rand_like(t0)
        t3 = t2 + t1
    def forward(self):
     ...    
# Input tensor
dummy_input = torch.randn(1, 2, 2)
