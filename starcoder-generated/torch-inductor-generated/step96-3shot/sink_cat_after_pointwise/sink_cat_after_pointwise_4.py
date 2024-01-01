
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inp1, inp2):
        t1 = torch.cat([inp1, inp2], dim=0)
        t2 = torch.cat([inp1, inp2], dim=0)
        x = torch.relu(t1 + t2)
        return x
# Inputs to the model
inp1 = torch.randn(1, 3, requires_grad=True)
inp2 = torch.randn(1, 4, requires_grad=True)
