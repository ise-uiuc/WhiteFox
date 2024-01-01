
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64, bias=True)
        self.l1 = torch.nn.Linear(64, 64, 3, 1, 1, bias=False)
        self.l2 = torch.add
    
    def forward(self, input_tensor)
        l1_out = self.linear(input_tensor)
        l1_out = self.l1(l1_out)
        l1_out = torch.clamp_min(l1_out, 0)
        l1_out = torch.clamp_max(l1_out, 6)
        l1_out = l1_out / 6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
