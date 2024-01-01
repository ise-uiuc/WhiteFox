
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, __input1__, __input2__):
        v1 = torch.cat([__input1__, __input2__], dim=1)
        v2 = v1[:, __input1__.size(1) : -1]
        v3 = v2[:, 0:__input2__.size(1)]
        v4 = torch.cat([__input1__, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
__input1__ = torch.randn(1, 2305843009213693952, 196608)
__input2__ = torch.randn(1, 2305843009213693952, 1048576)
