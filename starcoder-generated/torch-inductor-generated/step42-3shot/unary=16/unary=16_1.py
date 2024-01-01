
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(3, 2)
 
    def forward(self, x1):
        x2 = self.l1(torch.cat([x1, x1], dim=1))
        x3 = torch.nn.functional.gptj_gelu(x2)
        x4 = torch.nn.functional.gptj_gelu(x2)
        x5 = torch.nn.functional.gptj_gelu(x2)
        x6 = torch.nn.functional.gptj_gelu(x2)
        x7 = torch.nn.functional.gptj_gelu(x2)
        x8 = torch.nn.functional.gptj_gelu(x2)
        x9 = torch.nn.functional.gptj_gelu(x2)
        x10 = torch.nn.functional.gptj_gelu(x2)
        x11 = torch.nn.functional.gptj_gelu(x2)
        return x3, x4, x5, x6, x7, x8, x9, x10, x11

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
