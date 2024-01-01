
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9):
        q = self.multihead_attention(query=x1, key=x2, value=x3)
        # 1, 22, 256
        l = self.multihead_attention(query=q, key=x4, value=x5)
        # 1, 22, 108
        m = self.multihead_attention(query=l, key=x6, value=x7)
        # 1, 22, 54        
        n = self.multihead_attention(query=m, key=x8, value=x9)
        # 1, 22, 27
        return n


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.rand(1, 16, 256)
# |x1| = (1, 16, 256)
x2 = torch.rand(1, 24, 256)
# |x2| = (1, 24, 256)
x3 = torch.rand(1, 24, 256)
# |x3| = (1, 24, 256)
x4 = torch.rand(1, 28, 108)
# |x4| = (1, 28, 108)
x5 = torch.rand(1, 28, 108)
# |x5| = (1, 28, 108)
x6 = torch.rand(1, 32, 54)
# |x6| = (1, 32, 54)
x7 = torch.rand(1, 32, 54)
# |x7| = (1, 32, 54)
x8 = torch.rand(1, 36, 27)
# |x8| = (1, 36, 27)
x9 = torch.rand(1, 36, 27)
# |x9| = (1, 36, 27)
m(x1, x2, x3, x4, x5, x6, x7, x8, x9)

