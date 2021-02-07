import os
import shutil
import random

# 这种手动添加小的数据集后，记得修改clothing1m中的make dataset方法，不要random了，要使用原始的sort
# 取消dataset方法中，限制数量的部分；也就是这部分参数放在这部分处理；

def generate_small_clean_trainset(sample_per_class_clean_train):

    # small clean trainset
    root = "/sharedir/dataset"
    big_clean_trainset_path = os.path.join(root, "clean_train")
    small_clean_trainset_path = os.path.join(root,str(sample_per_class_clean_train)+'_small_clean_train')
    # 不能够删除，因为多个代码并行运行，删除一个可能会导致另外一个无法加载数据
    # if os.path.exists(small_clean_trainset_path):
    #     shutil.rmtree(small_clean_trainset_path)
    # os.makedirs(small_clean_trainset_path)
    for i in range(14):
        class_root_small = os.path.join(small_clean_trainset_path,str(i))
        os.makedirs(class_root_small)

        class_root_big = os.path.join(big_clean_trainset_path,str(i))
        for root, _, fnames in sorted(os.walk(class_root_big, followlinks=True)):
            random.shuffle(fnames)
            for j in range(sample_per_class_clean_train):
                shutil.copy(os.path.join(class_root_big,fnames[j]),os.path.join(class_root_small,fnames[j]))


def generate_small_clean_valset(sample_per_class_clean_val):

    # small clean trainset
    root = "/sharedir/dataset"
    big_clean_valset_path = os.path.join(root, "clean_val")
    small_clean_valset_path = os.path.join(root,str(sample_per_class_clean_val)+'_small_clean_val')
    # 不能够删除，因为多个代码并行运行，删除一个可能会导致另外一个无法加载数据
    # if os.path.exists(small_clean_trainset_path):
    #     shutil.rmtree(small_clean_trainset_path)
    # os.makedirs(small_clean_trainset_path)
    for i in range(14):
        class_root_small = os.path.join(small_clean_valset_path,str(i))
        os.makedirs(class_root_small)

        class_root_big = os.path.join(big_clean_valset_path,str(i))
        for root, _, fnames in sorted(os.walk(class_root_big, followlinks=True)):
            random.shuffle(fnames)
            for j in range(sample_per_class_clean_val):
                shutil.copy(os.path.join(class_root_big,fnames[j]),os.path.join(class_root_small,fnames[j]))



# generate_small_clean_trainset(sample_per_class_clean_train=10)
generate_small_clean_valset(sample_per_class_clean_val=10)

        


