
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat([x.unsqueeze(2), x.unsqueeze(1)], dim=0)
        x = x.view(-1, *x.shape[2:]) if x.shape == (2, 12) else x.view(-1, *x.shape[3:])
        if x.shape!= (3, 4, 3, 4):
            x = x.transpose(1, 3)
        x = x.mean(1)
        return x
# Inputs to the model
x = torch.randn(2, 2, 3, 4)
