
class ReLU6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.clamp(input, min=0, max=6)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (output,) = ctx.saved_tensors
        grad_input = grad_output.masked_fill(output > 6, 0.0)
        grad_input = grad_input.masked_fill(output < 0, 0.0)
        return grad_input

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        b = torch.cat((x1, x2),  1)
        a = ReLU6.apply(b)
        c = a[:, 0:4096]
        return c

# Initializing the model
model = Model()

# Inputs to the model
x1 = torch.randn(6, 4096, 10)
x2 = torch.randn(6, 4096, 20)
