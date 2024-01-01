t1 = input_tensor_A.permute(...) # Permute the input tensor A
t2 = input_tensor_B.permute(...) # Permute the input tensor B
t3 = torch.bmm(t1, t2) # or torch.matmul(t1, t2)
t1 = input_tensor_A.permute(...) # Permute the input tensor A
t2 = torch.bmm(t1, input_tensor_B) # or torch.matmul(t1, input_tensor_B)
t1 = input_tensor_B.permute(...) # Permute the input tensor B
t2 = torch.bmm(input_tensor_A, t1) # or torch.matmul(input_tensor_A, t1)
