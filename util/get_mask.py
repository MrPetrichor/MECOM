import numpy as np
from numpy.random import randint
import torch



def get_mask(view_num, alldata_len, missing_rate):
    nums = np.ones((view_num, alldata_len))
    nums[:, :int(alldata_len * missing_rate)] = 0
    for i in range (view_num):
        np.random.shuffle(nums[i])

    count = np.sum(nums, axis=0)
    add=0
    for i in range (alldata_len):
        if(count[i]==0):
            nums[randint(0,view_num)][i]=1
            add+=1
    dele=0
    one=0
    count = np.sum(nums, axis=0)
    for i in range (alldata_len):
        if(add==dele):
            break;
        if(count[i]>1):
            bb=randint(0, view_num )
            nums[bb][i]=0
            dele+=1
            one+=bb

    nums = torch.from_numpy(nums)


    return nums.to(torch.float32)

def get_element_mask(view_num, alldata_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    :param view_num:view number
    :param alldata_len:number of samples
    :param missing_rate:Defined in section 3.2 of the paper
    :return:Sn
    """
    nums = np.ones((view_num, alldata_len,2048))
    nums[:, :, :int(2048 * missing_rate)] = 0
    for i in range(view_num):
        for j in range(alldata_len):
            np.random.shuffle(nums[i][j])
    nums = torch.from_numpy(nums)

    return nums.to(torch.float32)

def get_mask_image(view_num, alldata_len, missing_rate):
    nums = np.ones((view_num, alldata_len))
    nums[0, :] = 0
    nums = torch.from_numpy(nums)
    return nums.to(torch.float32)

def get_mask_audio(view_num, alldata_len, missing_rate):
    nums = np.ones((view_num, alldata_len))
    nums[1, :] = 0
    nums = torch.from_numpy(nums)
    return nums.to(torch.float32)

def get_mask_text(view_num, alldata_len, missing_rate):
    nums = np.ones((view_num, alldata_len))
    nums[2, :] = 0
    nums = torch.from_numpy(nums)
    return nums.to(torch.float32)

def get_mask_video(view_num, alldata_len, missing_rate):
    nums = np.ones((view_num, alldata_len))
    nums[3, :] = 0
    nums = torch.from_numpy(nums)
    return nums.to(torch.float32)

if __name__ == '__main__':
    print(1)
