
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.0
 
    def forward(self, __input_tensor1__, __input_tensor2__, __input_tensor3__):
        matrix1 = torch.matmul(__input_tensor1__, __input_tensor2__.transpose(-2, -1))
        constant = 2.302585092994046
        inv_scale_factor = 1 / math.pow(constant, 1.3099999673706055)
        v1 = matrix1.div(inv_scale_factor)
        v2 = v1.softmax(dim=-1)
        v3 = torch.nn.functional.dropout(v2, p=self.dropout_p)
        v4 = v3.matmul(__input_tensor3__)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
__input_tensor1__ = torch.randn(1, 1, 4, 5)
__input_tensor2__ = torch.randn(1, 3, 15, 10)
__input_tensor3__ = torch.randn(1, 17, 40, 20)
