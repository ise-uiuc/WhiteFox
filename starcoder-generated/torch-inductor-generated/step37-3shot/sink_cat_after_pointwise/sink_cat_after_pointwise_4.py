
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.squeeze(dim=-1)
        x = torch.max(x, dim=1)[0].unsqueeze(dim=1)
        x = torch.relu(x).squeeze(dim=-1)
        x.unsqueeze(dim=-1)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
