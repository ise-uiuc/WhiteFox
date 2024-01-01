
class Model(torch.nn.Module):
    def forward(self, matrix1, matrix2, matrix3, matrix4):
        mm1 = torch.nn.functional.linear(matrix1, torch.randn_like(matrix1))
        mm_t = torch.mm(mm1, matrix2)
        t = mm_t + matrix3
        m1 = torch.nn.functional.linear(matrix1, torch.randn_like(matrix1))
        m2 = torch.mm(m1, matrix4)
        return t.matmul(m2)
# Inputs to the model
matrix1 = torch.randn(1, 3, 10)
matrix2 = torch.randn(5, 10)
matrix3 = torch.randn(1, 5)
matrix4 = torch.randn(5, 5)
