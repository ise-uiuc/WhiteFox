
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, mask): 
        o_mask = torch.nn.functional.dropout(mask, p=0.1)
        x = x * o_mask
        x = self.lin1(x)
        o_mask = torch.nn.Dropout(0.1)(o_mask)
        x = x * o_mask
        return x
# Inputs to the model
x = torch.randn(10, 5)
mask = torch.ones_like(x)
