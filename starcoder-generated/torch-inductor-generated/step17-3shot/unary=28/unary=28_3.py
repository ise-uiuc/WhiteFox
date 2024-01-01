
class Model(torch.nn.Module):
    def __init__(self, min_value=0, max_value=1):
        super().__init__()
 
    def forward(self, x1):
        v0 = x1 * 0.8959889827464012
        v1 = x1 + x1
        v2 = torch.mean(v1, dim=[-2, -1], keepdim=True)
        v3 = v1 / v2
        v4 = v3 * 0.3452390219632011
        v5 = torch.min(v4, torch.tensor(0.760402547351158, requires_grad=True), dim=[-2, -1], keepdim=True)
        v6 = torch.max(v5, torch.tensor(-0.426109205402870, requires_grad=True), dim=[-2, -1], keepdim=True)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 48, 48)
