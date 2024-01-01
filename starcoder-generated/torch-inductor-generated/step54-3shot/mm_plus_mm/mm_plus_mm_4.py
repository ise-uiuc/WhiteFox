
class Model(torch.nn.Module):
    def forward(self, input):
        y = torch.mm(input, input)   # 1st linear
        t1 = y + y                # 2nd linear and add
        t2 = y + y + y            # 3rd linear
        t3 = t1 * t2 + t2         
        return t3
# Inputs to the model
input = torch.randn(50, 50)
