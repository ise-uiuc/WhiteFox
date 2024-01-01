
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = B.Attention()
 
    def forward(self, x1, x2):
        v1 = self.attn(query=x1, key=x2, value=x2, dropout_p=0.1, scale_factor=4)
        return v1

# Generating a simple example to initialize the model
x1 = B.randn((8, 16, 16), 'float16')
x2 = B.randn((8, 16, 32), 'float16')
m = Model()
