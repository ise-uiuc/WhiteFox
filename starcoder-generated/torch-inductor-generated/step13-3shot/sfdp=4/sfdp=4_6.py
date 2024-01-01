
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        x = x1 @ x2.transpose(-2, -1).float()
        x = x / math.sqrt(x.size(-1)).float()
        x = x.to(torch.float32)
        if attn_mask:
            x = x + attn_mask.to(torch.float32)
        x = torch.softmax(x.to(torch.float32), dim=-1)
        x = x @ value
        return x

# Initializing the model
m = Model()
# Inputs to the model
x1 = torch.randn(2, 128, 512)
x2 = torch.randn(2, 512, 128)
mask = torch.tril(<mask_float_tensor>)
