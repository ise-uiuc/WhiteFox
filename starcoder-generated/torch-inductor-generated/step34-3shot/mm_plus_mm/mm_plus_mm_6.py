
class Model(torch.nn.Module):
    def forward(self, x):
        x = torch.mm(x, torch.ones(x.shape[-1], device=x.device).unsqueeze(-1))
        return x
# Inputs to the model
x = torch.ones(2,3)
