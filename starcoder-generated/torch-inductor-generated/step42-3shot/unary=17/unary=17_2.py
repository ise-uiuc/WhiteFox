
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten_module = nn.Flatten()
        self.linear_module = nn.Linear(16 * 7 * 7, 50)
        self.leakyrelu_module = nn.LeakyReLU()
        self.linear_module_1 = nn.Linear(50, 20)
        self.linear_module_2 = nn.Linear(20, 8)
        self.linear_module_3 = nn.Linear(8, 2)
        self.softmax_module = nn.Softmax()
    def forward(self, images):
        flatten_res = self.flatten_module(images)
        linear_res = self.linear_module(flatten_res)
        relu_res = self.leakyrelu_module(linear_res)
        linear_res = self.linear_module_1(relu_res)
        linear_res = self.linear_module_2(linear_res)
        linear_res = self.linear_module_3(linear_res)
        sm_res = self.softmax_module(linear_res)
        return sm_res
# Inputs to the model
x1 = torch.randn(1, 28, 28)
