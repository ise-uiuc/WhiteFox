
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, __input1__, __input2__):
        inv_scale = math.sqrt(__input2__.shape[-1])
        