import torch

Weights = torch.tensor([
    [  # First 2D slice (2 slices)
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ],
    [
        [13, 14, 15, 16],
        [17, 18, 19, 20],
        [21, 22, 23, 24]
    ] 
], dtype=torch.float32).permute([1,2,0])  # Shape: (2, 3, 4) "SOI"  permute to -> (Out_channels, In_channels, Kernel_size)

Input = torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20]
], dtype=torch.float32).permute([1,0]).unsqueeze(0)  # Shape: (5, 4) "SC" permute to -> (Batch, channels, width)

Bias = torch.tensor([1, 2, 3], dtype=torch.float32)  # Shape: (3,)

res = torch.nn.functional.conv1d(Input, Weights, Bias, stride=1, padding=0)

print(res.shape)
print(res)

# Output:
# torch.Size([1, 3, 4])
# tensor([[[ 413.,  685.,  957., 1229.],
#          [ 558.,  958., 1358., 1758.],
#          [ 703., 1231., 1759., 2287.]]])

# CMP with C++:
#                  C ->
#         413,      558,      703, 
# S       685,      958,     1231, 
# |       957,     1358,     1759, 
# V      1229,     1758,     2287, 



# test conv2d
# Matrix<int, "WHOI",3,3,2,3> Weights;
# Matrix<int, "WHC",5,6,3> Input;
# Matrix<int, "C",2> Bias{1,2};
Weights = torch.tensor(range(1,3*3*2*3+1)).view(3,3,2,3).float().permute([2,3,0,1])  # Shape: (3, 3, 2, 3) "WHOI" permute to -> (Out_channels, In_channels, Kernel_height, Kernel_width)
Input = torch.tensor(range(1,5*6*3+1)).view(5,6,3).float().permute([2,0,1]).unsqueeze(0)  # Shape: (5, 5, 3) "WHC" permute to -> (Batch, channels, height, width)
Bias = torch.tensor([1,2], dtype=torch.float32)  # Shape: ( 2,)

res = torch.nn.functional.conv2d(Input, Weights, Bias, stride=1, padding=0)
print(res.shape)
print(res)

# Output:
# torch.Size([1, 2, 3, 4])
# tensor([[[[22321., 24427., 26533., 28639.],
#           [34957., 37063., 39169., 41275.],
#           [47593., 49699., 51805., 53911.]],

#          [[24185., 26534., 28883., 31232.],
#           [38279., 40628., 42977., 45326.],
#           [52373., 54722., 57071., 59420.]]]])
# Cpp output:
# C = 0:
#                  W ->
#       22321,    34957,    47593, 
# H     24427,    37063,    49699, 
# |     26533,    39169,    51805, 
# V     28639,    41275,    53911, 
# C = 1:
#                  W ->
#       24185,    38279,    52373, 
# H     26534,    40628,    54722, 
# |     28883,    42977,    57071, 
# V     31232,    45326,    59420, 




# test conv2d stride:
# Matrix<int, "WHOI",3,3,2,3> Weights;
# Matrix<int, "WHC",5,6,3> Input;
# Matrix<int, "C",2> Bias{1,2};
Weights = torch.tensor(range(1,3*3*2*3+1)).view(3,3,2,3).float().permute([2,3,0,1])  # Shape: (3, 3, 2, 3) "WHOI" permute to -> (Out_channels, In_channels, Kernel_height, Kernel_width)
Input = torch.tensor(range(1,5*6*3+1)).view(5,6,3).float().permute([2,0,1]).unsqueeze(0)  # Shape: (5, 5, 3) "WHC" permute to -> (Batch, channels, height, width)
Bias = torch.tensor([1,2], dtype=torch.float32)  # Shape: ( 2,)

res = torch.nn.functional.conv2d(Input, Weights, Bias, stride=[2,1], padding=0)
print(res.shape)
print(res)

# Output:
# torch.Size([1, 2, 2, 4])
# tensor([[[[22321., 24427., 26533., 28639.],
#           [47593., 49699., 51805., 53911.]],

#          [[24185., 26534., 28883., 31232.],
#           [52373., 54722., 57071., 59420.]]]])

# Cpp output:
# Matrix of type: PermutedMatrix<"CWH", Matrix<int32_t, "WHC",  2,  4,  2>&>
# Order: DimensionOrder of length: 3 with content: "CWH"
# Dimensions: {2, 2, 4, }
# Data:
# C = 0:
#             W ->
#       22321,    47593, 
# H     24427,    49699, 
# |     26533,    51805, 
# V     28639,    53911, 

# C = 1:
#             W ->
#       24185,    52373, 
# H     26534,    54722, 
# |     28883,    57071, 
# V     31232,    59420, 


# test conv2d dilation:
# Matrix<int, "WHOI",3,3,2,3> Weights;
# Matrix<int, "WHC",7,6,3> Input;
# Matrix<int, "C",2> Bias{1,2};
Weights = torch.tensor(range(1,3*3*2*3+1)).view(3,3,2,3).float().permute([2,3,0,1])  # Shape: (3, 3, 2, 3) "WHOI" permute to -> (Out_channels, In_channels, Kernel_height, Kernel_width)
Input = torch.tensor(range(1,7*6*3+1)).view(7,6,3).float().permute([2,0,1]).unsqueeze(0)  # Shape: (5, 5, 3) "WHC" permute to -> (Batch, channels, height, width)
Bias = torch.tensor([1,2], dtype=torch.float32)  # Shape: ( 2,)

res = torch.nn.functional.conv2d(Input, Weights, Bias, stride=1, padding=0, dilation=[2,1])
print(res.shape)
print(res)

# Output:
# torch.Size([1, 2, 3, 4])
# tensor([[[[40789., 42895., 45001., 47107.],
#           [53425., 55531., 57637., 59743.],
#           [66061., 68167., 70273., 72379.]],

#          [[44111., 46460., 48809., 51158.],
#           [58205., 60554., 62903., 65252.],
#           [72299., 74648., 76997., 79346.]]]])

# Cpp output:
# Matrix of type: PermutedMatrix<"CWH", Matrix<int32_t, "WHC",  3,  4,  2>&>
# Order: DimensionOrder of length: 3 with content: "CWH"
# Dimensions: {2, 3, 4, }
# Data:
# C = 0:
#                  W ->
#       40789,    53425,    66061, 
# H     42895,    55531,    68167, 
# |     45001,    57637,    70273, 
# V     47107,    59743,    72379, 

# C = 1:
#                  W ->
#       44111,    58205,    72299, 
# H     46460,    60554,    74648, 
# |     48809,    62903,    76997, 
# V     51158,    65252,    79346, 

