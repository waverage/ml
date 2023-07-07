import torch

# tensor_one = torch.tensor([
#     [1, 2, 3],
# ])

# tensor_two = torch.tensor([
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9]
# ])

# print(tensor_one * tensor_two)
# print(torch.matmul(tensor_one, tensor_two))
# print(1 * 1 + 2 * 4 + 3 * 7, 1 * 2 + 2 * 5 + 3 * 8, 1 * 3 + 2 * 6 + 3 * 9)

tensor = torch.rand(3, 2)
print(torch.rand(3, 2))
print(torch.rand(2, 3))
print(torch.rand(3, 2) @ torch.rand(2, 3))