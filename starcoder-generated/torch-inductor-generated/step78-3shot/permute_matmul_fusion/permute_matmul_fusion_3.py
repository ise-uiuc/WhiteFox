
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_tensor_A_permute1 = torch.nn.functional.permute(input_tensor_A, [0, 2, 1])
        self.input_tensor_A_view1 = self.input_tensor_A_unsqueeze1.view([-1])
        self.input_tensor_B_permute1 = torch.nn.functional.permute(input_tensor_B, [0, 2, 1])
        self.input_tensor_B_view1 = self.input_tensor_B_unsqueeze1.view([-1])
        self.mat_mul1 = torch.nn.functional.linear(self.input_tensor_A_view1, self.input_tensor_B_view1)
    [Optional: Additional model layers can be added following self.mat_mul1 here]
    def forward(self, x1, x2):
        self.input_tensor_A_permute1 = torch.nn.functional.permute(x1, [0, 2, 1])
        self.input_tensor_B_permute1 = torch.nn.functional.permute(x2, [0, 2, 1])
        self.input_tensor_A_view1 = self.input_tensor_A_permute1.view([-1])
        self.input_tensor_B_view1 = self.input_tensor_B_permute1.view([-1])
        self.mat_mul1 = torch.mm(self.input_tensor_A_view1, self.input_tensor_B_view1)
        [Additional layers added here]
        return self.mat_mul1
# Inputs to the model
x1 = torch.randn(1, 2, 3)
x2 = torch.randn(1, 3, 2)
