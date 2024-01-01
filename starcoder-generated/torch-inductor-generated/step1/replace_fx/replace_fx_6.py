 
class CustomLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, x3):
        ctx.save_for_backward(x1, x3)
        return x1 + x2 + x3

    @staticmethod
    def backward(ctx, dy):
        dx1, dx3 = ctx.saved_tensors
        return dy, dy, dy

class CustomModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.custom_linear = CustomLinearFunction.apply

    def forward(self, x1):

        # Add Dropout here, call dropout twice, verify the output of the first
        # call is used in the second call.
        x1 = self.dropout(x1)
        x1 = self.dropout(x1)

        # Add a custom call to the linear function, verify the output of
        # torch.nn.functional.linear is used here.
        x2 = self.custom_linear(x1, self.linear1.weight, self.linear1.bias)
        x3 = torch.nn.functional.linear(x2, self.linear2.weight, self.linear2.bias)

        return x3

m = CustomModule()
dummy_input = torch.randn(1, 2, 2)
