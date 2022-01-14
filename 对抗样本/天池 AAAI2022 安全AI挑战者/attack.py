import torch
import numpy as np
import torch.nn as nn
from advertorch.utils import normalize_by_pnorm, batch_multiply, batch_clamp
import copy


def pgd_attack(model1,
               model2,
               X,
               y,
               epsilon=0.03,
               clip_max=1.0,
               clip_min=0.0,
               num_steps=50,
               step_size=0.01,
               print_process=False,
               bound='linf'):
    # out = model(X)
    # err = (out.data.max(1)[1] != y.data).float().sum()

    # TODO: find a other way
    device = X.device
    imageArray = X.detach().cpu().numpy()
    X_random = np.random.uniform(-epsilon, epsilon, X.shape)
    imageArray = np.clip(imageArray + X_random, 0, 1.0)

    X_pgd = torch.tensor(imageArray).to(device).float()
    X_pgd.requires_grad = True

    for i in range(num_steps):

        pred1 = model1(X_pgd)
        loss1 = nn.CrossEntropyLoss()(pred1, y)
        pred2 = model2(X_pgd)
        loss2 = nn.CrossEntropyLoss()(pred2, y)
        loss = loss1 + loss2
        # pred1 = model1(X_pgd)
        # pred2 = model2(X_pgd)
        # pred=(pred1+pred2)/2
        # loss = nn.CrossEntropyLoss()(pred2, y)

        # print("11",pred1.data.max(1)[1])
        # print("22", pred2.data.max(1)[1])
        # print("33",y.data)
        # print(err)

        if print_process:
            print("iteration {:.0f}, loss:{:.4f}".format(i, loss))

        loss.backward()

        if bound == 'linf':
            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = X_pgd + eta
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)

            X_pgd = X.data + eta

            # print(X_pgd.size())

            X_pgd = X_pgd.permute(1, 0, 2, 3)
            X_pgd[0].clamp(max=(1 - 0.4914) / 0.2023, min=(0 - 0.4914) / 0.2023)
            X_pgd[1].clamp(max=(1 - 0.4822) / 0.1994, min=(0 - 0.4822) / 0.1994)
            X_pgd[2].clamp(max=(1 - 0.4465) / 0.2010, min=(0 - 0.4465) / 0.2010)
            X_pgd = X_pgd.permute(1, 0, 2, 3)

            # X_pgd = torch.clamp(X_pgd, clip_min, clip_max)
            # for ind in range(X_pgd.shape[1]):
            #    X_pgd[:,ind,:,:] = (torch.clamp(X_pgd[:,ind,:,:] * std[ind] + mean[ind], clip_min, clip_max) - mean[ind]) / std[ind]

            X_pgd = X_pgd.detach()
            X_pgd.requires_grad_()
            X_pgd.retain_grad()

        # if bound == 'l2':
        #     output = model(X + delta)
        #     incorrect = output.max(1)[1] != y
        #     correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
        #     # Finding the correct examples so as to attack only them
        #     loss = nn.CrossEntropyLoss()(model(X + delta), y)
        #     loss.backward()
        #     delta.data += correct * alpha * delta.grad.detach() / norms(delta.grad.detach())
        #     delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
        #     delta.data = torch.min(torch.max(delta.detach(), -X), 1 - X)  # clip X+delta to [0,1]
        #     delta.grad.zero_()

    with torch.no_grad():
        pred1 = model1(X_pgd)
        pred2 = model2(X_pgd)
        err1 = (pred1.data.max(1)[1] != y.data).float().sum().item()
        err2 = (pred2.data.max(1)[1] != y.data).float().sum().item()
        print(err1, err2)

    return X_pgd, err1, err2


def fgsm(model1, model2, image, label, epsilon=0.2):
    imageArray = image.cpu().detach().numpy()
    device = image.device
    X_fgsm = torch.tensor(imageArray).to(device)

    # print(image.data)
    X_fgsm.requires_grad = True

    # opt = optim.SGD([X_fgsm], lr=1e-3)
    # opt.zero_grad()

    loss1 = nn.CrossEntropyLoss()(model1(X_fgsm), label)
    loss2 = nn.CrossEntropyLoss()(model2(X_fgsm), label)
    loss = loss1 + loss2

    loss.backward()
    # print(X_fgsm)
    # print(X_fgsm.grad)
    # if order == np.inf:
    d = epsilon * X_fgsm.grad.data.sign()
    # elif order == 2:
    #     gradient = X_fgsm.grad
    #     d = torch.zeros(gradient.shape, device=device)
    #     for i in range(gradient.shape[0]):
    #         norm_grad = gradient[i].data / LA.norm(gradient[i].data.cpu().numpy())
    #         d[i] = norm_grad * epsilon
    # else:
    #     raise ValueError('Other p norms may need other algorithms')

    x_adv = X_fgsm + d

    x_adv = x_adv.permute(1, 0, 2, 3)
    x_adv[0].clamp(max=(1 - 0.4914) / 0.2023, min=(0 - 0.4914) / 0.2023)
    x_adv[1].clamp(max=(1 - 0.4822) / 0.1994, min=(0 - 0.4822) / 0.1994)
    x_adv[2].clamp(max=(1 - 0.4465) / 0.2010, min=(0 - 0.4465) / 0.2010)
    x_adv = x_adv.permute(1, 0, 2, 3)

    with torch.no_grad():
        pred1 = model1(x_adv)
        pred2 = model2(x_adv)
        err1 = (pred1.data.max(1)[1] != label.data).float().sum().item()
        err2 = (pred2.data.max(1)[1] != label.data).float().sum().item()
        # print(err1, err2)

    return x_adv, err1, err2


def MIM(model1, model2, eps=0.3, nb_iter=40, decay_factor=0.8,
        eps_iter=0.01, clip_min=0., clip_max=1., targeted=False, ord=np.inf, x=None, y=None):
    eps = eps
    nb_iter = nb_iter
    decay_factor = decay_factor
    eps_iter = eps_iter
    targeted = targeted
    ord = ord

    delta = torch.zeros_like(x)
    g = torch.zeros_like(x)

    delta = nn.Parameter(delta, requires_grad=True)

    for i in range(nb_iter):
        # print(i)
        # if delta.grad is not None:
        #     delta.grad.detach_()
        #     delta.grad.zero_()
        #     delta.retain_grad()

        delta = delta.detach()
        delta.requires_grad_()

        imgadv = x + delta

        output1 = model1(imgadv)
        output2 = model2(imgadv)

        loss1 = nn.CrossEntropyLoss()(output1, y)
        loss2 = nn.CrossEntropyLoss()(output2, y)
        loss = loss1 + loss2
        loss.backward()

        g = decay_factor * g + normalize_by_pnorm(delta.grad.data, p=1)
        # according to the paper it should be .sum(), but in their
        #   implementations (both cleverhans and the link from the paper)
        #   it is .mean(), but actually it shouldn't matter
        if ord == np.inf:
            delta.data += batch_multiply(eps_iter, torch.sign(g))
            delta.data = batch_clamp(eps, delta.data)

            # print(delta.size())

            delta = delta.permute(1, 0, 2, 3)
            delta[0].clamp(max=(1 - 0.4914) / 0.2023, min=(0 - 0.4914) / 0.2023)
            delta[1].clamp(max=(1 - 0.4822) / 0.1994, min=(0 - 0.4822) / 0.1994)
            delta[2].clamp(max=(1 - 0.4465) / 0.2010, min=(0 - 0.4465) / 0.2010)
            delta = delta.permute(1, 0, 2, 3)
        elif ord == 2:
            delta.data += eps_iter * normalize_by_pnorm(g, p=2)
            delta.data *= torch.clamp((eps * normalize_by_pnorm(delta.data, p=2) / delta.data), max=1.)

            delta = delta.permute(1, 0, 2, 3)
            delta[0].clamp(max=(1 - 0.4914) / 0.2023, min=(0 - 0.4914) / 0.2023)
            delta[1].clamp(max=(1 - 0.4822) / 0.1994, min=(0 - 0.4822) / 0.1994)
            delta[2].clamp(max=(1 - 0.4465) / 0.2010, min=(0 - 0.4465) / 0.2010)
            delta = delta.permute(1, 0, 2, 3)
        else:
            error = "Only ord = inf and ord = 2 have been implemented"
            raise NotImplementedError(error)

    rval = x + delta.data

    with torch.no_grad():
        pred1 = model1(rval)
        pred2 = model2(rval)
        err1 = (pred1.data.max(1)[1] != y.data).float().sum().item()
        err2 = (pred2.data.max(1)[1] != y.data).float().sum().item()
    return rval, err1, err2


def deepfool(model1, model2, image, num_classes=10, overshoot=0.02, max_iter=50):
    assert image.size()[0] == 1

    f_image1 = model1(image).data.cpu().numpy().flatten()
    f_image2 = model2(image).data.cpu().numpy().flatten()

    # 获取由大到小的索引
    output_resnet = np.array(f_image1).argsort()[::-1]
    output_densenet = np.array(f_image2).argsort()[::-1]
    # 获取模型预测label
    label_resnet = output_resnet[0]
    label_densenet = output_densenet[0]

    input_shape = image.cpu().numpy().shape
    x = copy.deepcopy(image).requires_grad_(True)

    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    fs1 = model1(x)
    fs1_list = list(torch.squeeze(fs1))
    current_pred_label_resnet = label_resnet
    fs2 = model2(x)
    fs2_list = list(torch.squeeze(fs2))
    current_pred_label_densenet = label_densenet

    for i in range(max_iter):
        pert = np.inf
        fs1[0, label_resnet].backward(retain_graph=True)
        fs2[0, label_densenet].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            # x.zero_grad()
            x.grad.zero_()
            fs1[0, output_resnet[k]].backward(retain_graph=True)
            fs2[0, output_densenet[k]].backward(retain_graph=True)

            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs1[0, output_resnet[k]] + fs2[0, output_densenet[k]]
                   - fs1[0, output_resnet[0]] - fs2[0, output_densenet[0]]).data.cpu().numpy()

            pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i = (pert + 1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot).cuda()

        pert_image[0][0].clamp(max=(1 - 0.4914) / 0.2023, min=(0 - 0.4914) / 0.2023)
        pert_image[0][1].clamp(max=(1 - 0.4822) / 0.1994, min=(0 - 0.4822) / 0.1994)
        pert_image[0][2].clamp(max=(1 - 0.4465) / 0.2010, min=(0 - 0.4465) / 0.2010)

        x = pert_image.detach().requires_grad_(True)
        fs1 = model1(x)
        fs2 = model2(x)

        if (np.argmax(fs1.data.cpu().numpy().flatten()) != label_resnet) and\
                (np.argmax(fs2.data.cpu().numpy().flatten()) != label_densenet):
            # print(i)
            break

    r_tot = (1 + overshoot) * r_tot

    return pert_image, r_tot, i
