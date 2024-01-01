
# Description: This model has the same pattern between 
#             the cat operations. The optimization 
#             should be triggered.
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(-1)
        x = torch.cat((y, y, y, y), dim=0)
        z = y.view(y.shape[0], -1)
        return z
# Inputs to the model
x = torch.randn(2, 8, 4)
