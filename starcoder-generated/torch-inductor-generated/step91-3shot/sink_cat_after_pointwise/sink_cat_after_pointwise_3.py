
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.tensor([0.56, 1.81], dtype=torch.float32, device=x.device)
        z = torch.cat((y, x), dim=1)
        w = z + 1 + 1
        if w.shape == (2, 20):
            z = z.t()
        #if z.shape == (2, 20):
        return w
# Inputs to the model
x = torch.randn((2, 5))
