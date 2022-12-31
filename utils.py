import torch

# define a Label Smoothing loss function

def CrossEntropyLoss_label_smooth(outputs, targets, num_classes=10, epsilon=0.1):
    N = targets.size(0)
    # 初始化一个矩阵, 里面的值都是epsilon / (num_classes - 1)
    smoothed_labels = torch.full(size=(N, num_classes), fill_value=epsilon / (num_classes - 1)).to(device)

    targets = targets.data
    # 为矩阵中的每一行的某个index的位置赋值为1 - epsilon
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(targets, dim=1), value=1 - epsilon)
    # 调用torch的log_softmax
    log_prob = nn.functional.log_softmax(outputs, dim=1)
    # 用之前得到的smoothed_labels来调整log_prob中每个值
    loss = - torch.sum(log_prob * smoothed_labels) / N
    return loss

# generate frac_vectors

def lateral_index(index, alpha):
    x = []
    for i in range(index):
        if i == 0:
            tmp = 1
        else:
            tmp = (1 - (alpha + 1) / i) * tmp
        x.append(tmp)
    return x


def frac_coeffients(alpha, n_class, label):
    x_1 = lateral_index(n_class - label, alpha)
    x_2 = lateral_index(label + 1, alpha)
    list.reverse(x_2)
    x = x_2 + x_1
    del x[label]

    return x


def generate_label(labels, alpha, n_class=10):
    a = []
    for i in range(len(labels)):
        x = frac_coeffients(alpha, n_class, labels[i])
        a.append(x)
    a = np.array(a)
    return torch.from_numpy(a).float()


# define mixup functions
def mixup_data(x, y):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    alpha = 0.5
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    labels = generate_label(y, alpha).to(device)
    mixed_labels = lam * labels + (1 - lam) * labels[index, :]
    target_a = y
    target_b = y[index]
    return mixed_x, mixed_labels, target_a, target_b, lam


def mixup_criterion(pred, y, criterion=simi_criterion):
    return torch.mean(criterion(pred, y))