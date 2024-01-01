
class Model(torch.nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.head_nets = torch.nn.ModuleList([torch.nn.Linear(128*4, 64) for _ in range]) 
    def forward(self, x):
        for i in range(num_heads):
            x = x + self.head_nets[i](x)
            x = torch.tanh(x)
        return x

# Inputs to the model
x = torch.randn(128*4, 32)
