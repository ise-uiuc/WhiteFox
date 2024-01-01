
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input, other=None, ):
        t1 = torch.nn.functional.max_pool2d(input, kernel_size=(3, 3), stride=1, padding=(2,2), ceil_mode=False, return_indices=False, dilation=1)
        if other == None:
            other = []
        t2 = t1 + other
        return t2
# Inputs to the model
input = torch.randn(2, 3, 64, 64)
