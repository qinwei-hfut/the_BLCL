from __future__ import print_function, absolute_import

__all__ = ['accuracy','compute_loader_label_acc','stack_params','visual_weight_grad']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# 计算dataset的noisy rate
def compute_loader_label_acc(dataloader):
    correct_label_count = 0.0
    total_count = 0.0
    for batch_idx, (inputs, targets, soft_targets, true_targets, indexs) in enumerate(dataloader):
        correct_label_count += (targets == true_targets).sum()
        total_count += targets.size(0)

    return correct_label_count / total_count

def observe_trainset(dataloader):
    for batch_idx, (inputs, targets, soft_targets, true_targets, indexs) in enumerate(dataloader):
        for i in range(targets.size()):
            print('-------')
            print(torch.argmax(soft_targets[i],dim=1))
            print(true_targets[i])
            print(soft_targets[i])


# 累加layer中的block的参数generator
def stack_params(module_list):
    for module in module_list:
        yield from module.params()

# 计算每一层的平均weight和平均grad
def visual_weight_grad(model,writer,training_iteration_idx):
    # key是每一层的name，value是一个tuple (每一层的参数量，参数梯度之和，参数之和)
    name_grad_num_dict = {}
    for name, params in model.named_params(model):
        name_grad_num_dict[name.split('.')[0]] = (0.0,0.0,0.0)
    for name, params in model.named_params(model):
        name_grad_num_dict[name.split('.')[0]] = (name_grad_num_dict[name.split('.')[0]][0]+params.numel(), name_grad_num_dict[name.split('.')[0]][1]+params.grad.sum(), name_grad_num_dict[name.split('.')[0]][2]+params.sum())
    

    keys_items = list(name_grad_num_dict.keys())

    # pdb.set_trace()
    # visual 每一层的梯度 grad平均值
    writer.add_scalars('grad',
        {keys_items[0]: name_grad_num_dict[keys_items[0]][1] / name_grad_num_dict[keys_items[0]][0],
        keys_items[1]: name_grad_num_dict[keys_items[1]][1] / name_grad_num_dict[keys_items[1]][0],
        keys_items[2]: name_grad_num_dict[keys_items[2]][1] / name_grad_num_dict[keys_items[2]][0],
        keys_items[3]: name_grad_num_dict[keys_items[3]][1] / name_grad_num_dict[keys_items[3]][0],
        keys_items[4]: name_grad_num_dict[keys_items[4]][1] / name_grad_num_dict[keys_items[4]][0],
        keys_items[5]: name_grad_num_dict[keys_items[5]][1] / name_grad_num_dict[keys_items[5]][0],   
    }, training_iteration_idx)

    # visual 每一层的参数平均值
    writer.add_scalars('weight',
        {keys_items[0]: name_grad_num_dict[keys_items[0]][2] / name_grad_num_dict[keys_items[0]][0],
        keys_items[1]: name_grad_num_dict[keys_items[1]][2] / name_grad_num_dict[keys_items[1]][0],
        keys_items[2]: name_grad_num_dict[keys_items[2]][2] / name_grad_num_dict[keys_items[2]][0],
        keys_items[3]: name_grad_num_dict[keys_items[3]][2] / name_grad_num_dict[keys_items[3]][0],
        keys_items[4]: name_grad_num_dict[keys_items[4]][2] / name_grad_num_dict[keys_items[4]][0],
        keys_items[5]: name_grad_num_dict[keys_items[5]][2] / name_grad_num_dict[keys_items[5]][0],   
    }, training_iteration_idx)


