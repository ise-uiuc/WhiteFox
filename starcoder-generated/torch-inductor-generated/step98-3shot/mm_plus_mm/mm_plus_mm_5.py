
class Model(torch.nn.Module):
    def __init__(self, mat_mul, add):
        super(Model, self).__init__()
        self.mat_mul = mat_mul
        self.add = add

    def forward(self, input):
        t1 = self.mat_mul(input, input)
        t2 = self.mat_mul(input, input)
        return self.add(t1, t2)
# Inputs to the model
input1 = torch.randn(5, 5)
