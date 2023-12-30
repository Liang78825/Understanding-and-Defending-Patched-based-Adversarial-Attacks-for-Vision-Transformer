import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from torchvision.utils import save_image
import numpy as np
import time
from pytorch_pretrained_vit import ViT
from utils import clamp, get_loaders, my_logger, my_meter, PCGrad
from pytorch_pretrained_vit.deit import deit_tiny_patch16_LS as DeiT


def get_aug():
    parser = argparse.ArgumentParser(description='Patch-Fool Training')

    parser.add_argument('--name', default='', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--dataset', default='ImageNet', type=str)
    parser.add_argument('--data_dir', default='/data1/ImageNet/ILSVRC/Data/CLS-LOC/', type=str)
    parser.add_argument('--log_dir', default='log', type=str)
    parser.add_argument('--crop_size', default=224, type=int)
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--workers', default=16, type=int)

    parser.add_argument('--network', default='DeiT-B', type=str, choices=['DeiT-B', 'DeiT-S', 'DeiT-T',
                                                                          'ResNet152', 'ResNet50', 'ResNet18'])
    parser.add_argument('--dataset_size', default=1.0, type=float, help='Use part of Eval set')

    parser.add_argument('--patch_select', default='Attn', type=str, choices=['Rand', 'Saliency', 'Attn'])
    parser.add_argument('--num_patch', default=1, type=int)
    parser.add_argument('--sparse_pixel_num', default=0, type=int)

    parser.add_argument('--attack_mode', default='CE_loss', choices=['CE_loss', 'Attention'], type=str)
    parser.add_argument('--atten_loss_weight', default=0.002, type=float)
    parser.add_argument('--atten_select', default=4, type=int, help='Select patch based on which attention layer')
    parser.add_argument('--mild_l_2', default=0., type=float, help='Range: 0-16')
    parser.add_argument('--mild_l_inf', default=0., type=float, help='Range: 0-1')

    parser.add_argument('--train_attack_iters', default=250, type=int)
    parser.add_argument('--random_sparse_pixel', action='store_true', help='random select sparse pixel or not')
    parser.add_argument('--learnable_mask_stop', default=200, type=int)

    parser.add_argument('--attack_learning_rate', default=0.22, type=float)
    parser.add_argument('--step_size', default=10, type=int)
    parser.add_argument('--gamma', default=0.95, type=float)

    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    args = parser.parse_args()

    if args.mild_l_2 != 0 and args.mild_l_inf != 0:
        print(f'Only one parameter can be non-zero: mild_l_2 {args.mild_l_2}, mild_l_inf {args.mild_l_inf}')
        raise NotImplementedError
    if args.mild_l_inf > 1:
        args.mild_l_inf /= 255.
        print(f'mild_l_inf > 1. Constrain all the perturbation with mild_l_inf/255={args.mild_l_inf}')

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    return args


global atten_grad


def extract_grad(score):
    atten_grad = score


def main():
    args = get_aug()

    logger = my_logger(args)
    meter = my_meter()
    is_deit = False
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    patch_size = 32
    filter = torch.ones([1, 3, patch_size, patch_size]).float().cuda()
    '''
    if args.network == 'ResNet152':
        model = ResNet152(pretrained=True)
    elif args.network == 'ResNet50':
        model = ResNet50(pretrained=True)
    elif args.network == 'ResNet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif args.network == 'VGG16':
        model = torchvision.models.vgg16(pretrained=True)
    elif args.network == 'DeiT-T':
        model = deit_tiny_patch16_224(pretrained=True)
    elif args.network == 'DeiT-S':
        model = deit_small_patch16_224(pretrained=True)
    elif args.network == 'DeiT-B':
        model = deit_base_patch16_224(pretrained=True)
    else:
        print('Wrong Network')
        raise
    '''
    if is_deit:
        model = DeiT(pretrained=True)
        layer_num = len(model.blocks)
    else:
        model = ViT('B_32_imagenet1k', pretrained=True)
        layer_num = model.transformer.num_layers
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    model.eval()

    list_tau = [[0]] * layer_num
    list_tau_2 = [[0]] * layer_num

    criterion = nn.CrossEntropyLoss().cuda(1)
    is_patch = True
    # eval dataset
    # loader = get_loaders(args)
    mu = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()

    train_dir = 'c:\\vit\\imagenet\\train'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    data_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=0)

    start_time = time.time()

    '''Original image been classified incorrect but turn to be correct after adv attack'''
    false2true_num = 0
    layer_log = torch.zeros(layer_num, 3)

    rr = torch.zeros(2, 12)
    rrr = torch.zeros(2, 12, 12)
    count = 1

    # if True:
    for i, (X, y) in enumerate(data_loader):
        '''not using all of the eval dataset to get the final result'''
        if i == int(len(data_loader) * args.dataset_size):
            break

        if i == 82:
            i = i
        '''
        X = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])(Image.open('/home/liang/vit/imagenet-sample-images-master/n01494475_hammerhead.JPEG')).unsqueeze(0)
        X = X.cuda()
        y  = torch.tensor([0]).long().cuda()
        '''
        X = X.cuda()
        y = y.cuda()
        num_patch = 5
        patch_num_per_line = int(X.size(-1) / patch_size)
        delta = torch.zeros_like(X).cuda()
        delta.requires_grad = True

        model.zero_grad()
        for l in range(layer_num):
            if is_deit:
                break
            else:
                #   model.module.transformer.blocks[l].attn.clean = 1

                model.module.transformer.blocks[l].attn.patch = None

        out0 = model(X + delta)
        classification_result = out0.max(1)[1] == y
        correct_num = classification_result.sum().item()
        losss = criterion(out0, y)
        meter.add_loss_acc("Base", {'CE': losss.item()}, correct_num, y.size(0))

        if not classification_result:
            print('batch ', i, ': skip')
            continue
        else:
            print('batch', i)

        # Skip Test
        if False:
            with torch.no_grad():
                out_list = torch.zeros(layer_num)
                aa = nn.Softmax(dim=1)
                with torch.no_grad():
                    for n in range(layer_num + 1):
                        for m in range(layer_num):
                            if m == n:
                                model.module.transformer.blocks[m].skip = True
                            else:
                                model.module.transformer.blocks[m].skip = False
                        out1 = aa(model(X))
                        if out1[0, y] < 0.5 and n != layer_num:
                            layer_log[n, 0] = layer_log[n, 0] + 1

                        if n != layer_num:
                            out_list[n] = out1[0, y]
                    print('skip of benign output', out_list)
        if is_deit is False:
            for l in range(layer_num):
                model.module.transformer.blocks[l].attn.clean = -1

        '''choose patch'''
        # max_patch_index size: [Batch, num_patch attack]
        if is_deit:
            atten_layer = torch.zeros(model.module.blocks[0].attn.scores.size(3) - 1).cuda()
            for l in range(11):
                atten_layer = atten_layer + model.module.blocks[l].attn.scores.mean(1).mean(1)[:, 1:]
        else:
            atten_layer = torch.zeros(model.module.transformer.blocks[0].attn.scores.size(3) - 1).cuda()
            for l in range(11):
                atten_layer = atten_layer + model.module.transformer.blocks[l].attn.scores.mean(1).mean(1)[:, 1:]

        # model.module.transformer.blocks[i].attn.scores.mean(1).mean(1)[0, 1:].view(12, 12)
        max_patch_index = atten_layer.argsort(descending=True)[:, :num_patch]

        '''build mask'''
        mask = torch.zeros([X.size(0), 1, X.size(2), X.size(3)]).cuda()
        # if args.sparse_pixel_num != 0:
        #   learnable_mask = mask.clone()
        for j in range(X.size(0)):
            index_list = max_patch_index[j]
            for index in index_list:
                row = (index // patch_num_per_line) * patch_size

                column = (index % patch_num_per_line) * patch_size
                mask[j, :, row:row + patch_size, column:column + patch_size] = 1

            if is_patch is False:
                mask = 1

        if is_deit is False and False:
            for l in range(12):
                model.module.transformer.blocks[l].attn.patch = max_patch_index + 1
                if i == 0:
                    aa = model.module.transformer.blocks[10].attn.scores[0].mean(0).mean(0).topk(40)[1]
                    aa = aa[aa % 12 > 3]
                    aa = aa[aa % 12 < 11]
                    aa = aa[aa > 12]
                    aa = aa[aa < 100]
                else:
                    aa = model.module.transformer.blocks[10].attn.scores[0].mean(0).mean(0).topk(50)[1]
                    aa = aa[aa % 12 > 1]
                    aa = aa[aa % 12 < 11]
                    aa = aa[aa > 12]
                    aa = aa[aa < 130]
                mm = torch.zeros(145, dtype=torch.bool).cuda()
                mm[aa + 1] = True
                model.module.transformer.blocks[l].attn.feature = mm

        '''adv attack'''
        max_patch_index_matrix = max_patch_index[:, 0]
        if is_deit:
            max_patch_index_matrix = max_patch_index_matrix.repeat(model.module.blocks[0].attn.scores.size(2), 1)
        else:
            max_patch_index_matrix = max_patch_index_matrix.repeat(
                model.module.transformer.blocks[0].attn.scores.size(2), 1)
            for l in range(layer_num):
                model.module.transformer.blocks[l].get_grad = max_patch_index[0, 0]
        max_patch_index_matrix = max_patch_index_matrix.permute(1, 0)
        max_patch_index_matrix = max_patch_index_matrix.flatten().long()

        if args.mild_l_inf == 0:
            '''random init delta'''
            delta = (torch.rand_like(X) - mu) / std
        else:
            '''constrain delta: range [x-epsilon, x+epsilon]'''
            epsilon = args.mild_l_inf / std
            delta = 2 * epsilon * torch.rand_like(X) - epsilon + X

        delta.data = clamp(delta, (0 - mu) / std, (1 - mu) / std)
        original_img = X.clone()

        X = torch.mul(X, 1 - mask)

        delta = delta.cuda()
        delta.requires_grad = True

        opt = torch.optim.Adam([delta], lr=args.attack_learning_rate)

        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.gamma)
        m = nn.Softmax(dim=1)
        abc = torch.zeros(layer_num, 250)
        '''Start Adv Attack'''
        for train_iter_num in range(args.train_attack_iters):
            model.zero_grad()
            opt.zero_grad()

            '''final CE-loss'''
            out = model(X + torch.mul(delta, mask))
            if False:
                if train_iter_num%10 == 1:
                    f=open('cc.txt','a')
                    for l in range(layer_num):
                        f.write(str(model.module.transformer.blocks[l].attn.scores[0].mean(0)[1:, 1:].mean(0)[max_patch_index].detach().cpu().numpy().tolist()[0]))
                        f.write(',')
                    f.write('\n')

            if is_deit is False:
                with torch.no_grad():
                    for l in range(layer_num):
                        abc[l, train_iter_num] = model.module.transformer.blocks[l].in_x_1[
                                                     0, max_patch_index[0, 0] + 1].norm() / \
                                                 model.module.transformer.blocks[l].in_x[
                                                     0, max_patch_index[0, 0] + 1].norm()

            # print(train_iter_num, out.topk(2)[0].tolist())
            loss = criterion(out, y)

            grad = torch.autograd.grad(loss, delta, retain_graph=True)[0]
            ce_loss_grad_temp = grad.view(X.size(0), -1).detach().clone()
            if train_iter_num %10 == 1:
                f=open('aa.txt','a')
                f.write(str(loss.detach().cpu().numpy().tolist()))
                f.write(',')

            # Attack the first 6 layers' Attn
            range_list = range(3)
            if is_patch is False:
                range_list = range(0)

            for atten_num in range_list:
                if atten_num == 0:
                    continue
                if is_deit:
                    atten_map = model.module.blocks[atten_num].attn.scores
                else:
                    atten_map = model.module.transformer.blocks[atten_num].attn.scores
                atten_map = atten_map.mean(dim=1)
                atten_map = atten_map.view(-1, atten_map.size(-1))

                atten_map = -torch.log(atten_map)

                atten_loss = F.nll_loss(atten_map, max_patch_index_matrix)

                if True:
                    for l in range(12):
                        a,pp = model.module.transformer.blocks[l].attn.scores[0, :, 1:,1:].mean(0).mean(0).topk(2)
                        if pp[0]==max_patch_index[0,0]:
                            p=pp[1].view(1,1)
                        else:
                            p=pp[0].view(1,1)
                        a = model.module.transformer.blocks[l].attn.scores[0, :, 1:,1:][:,:,max_patch_index[0]].mean()/model.module.transformer.blocks[l].attn.scores[0, :, 1:,1:][:,:,p[0]].mean()
                        if a > 1.4:
                            atten_loss -= 1000 * (model.module.transformer.blocks[l].attn.scores[0, :, 1:,1:][:,:,max_patch_index[0]]
                                               - 1.4 * model.module.transformer.blocks[l].attn.scores[0, :, 1:,1:][:,:,p[0]]).norm()
                        if train_iter_num %10 == 1 and atten_num==1:
                            f.write(str(a.detach().cpu().numpy().tolist()))
                            f.write(',')

                atten_grad = torch.autograd.grad(atten_loss, delta, retain_graph=True)[0]

                atten_grad_temp = atten_grad.view(X.size(0), -1)
                cos_sim = F.cosine_similarity(atten_grad_temp, ce_loss_grad_temp, dim=1)

                '''PCGrad'''
                atten_grad = PCGrad(atten_grad_temp, ce_loss_grad_temp, cos_sim, grad.shape)

                grad += atten_grad * args.atten_loss_weight

            opt.zero_grad()
            delta.grad = -grad
            opt.step()
            scheduler.step()

            epsilon = 1 / std

            if is_patch is False:
                epsilon = 0.1 / std

            delta.data = clamp(delta, original_img - epsilon, original_img + epsilon)

            delta.data = clamp(delta, (0 - mu) / std, (1 - mu) / std)

        if True:
            f.write('\n')
            f.close()
        '''Eval Adv Attack'''
        perturb_x = X + torch.mul(delta, mask)

        out2 = model(perturb_x)
        classification_result_after_attack = out2.max(1)[1] == y

        if False:
            with torch.no_grad():
                delta2 = torch.mul(delta, mask)
            delta2.requires_grad=True
            out2 = model(X + delta2)
            loss = criterion(out2, out2.max(1)[1])
            grad = torch.autograd.grad(loss,delta2)[0]
            slc, _ = torch.max(grad[0].abs(), dim=0)
            slc = (slc - slc.min()) / (slc.max() - slc.min())
            aa=torch.zeros(24,24)
            for x in range(24):
                for y in range(24):
                    aa[x,y] = slc[x:x+16,y:y+16].mean()

        if False:
            for l in range(12):
                rr[0, l] = rr[0, l] + (model.module.transformer.blocks[l].in_x_1[0, max_patch_index + 1].norm() /
                                       model.module.transformer.blocks[l].in_x[0, max_patch_index + 1].norm())

                rr[1, l] = rr[1, l] + (model.module.transformer.blocks[l].in_x_1[0].norm(dim=1) /
                                       model.module.transformer.blocks[l].in_x[0].norm(dim=1)).mean()

                for ll in range(12):
                    rrr[0, l, ll] = rrr[0, l, ll] + torch.cosine_similarity(
                        model.module.transformer.blocks[l].in_x[0, max_patch_index[0] + 1],
                        model.module.transformer.blocks[ll].in_x[0, max_patch_index[0] + 1])

                    rrr[1, l, ll] = rrr[1, l, ll] + torch.cosine_similarity(
                        model.module.transformer.blocks[l].in_x[0].mean(0).view(1, -1),
                        model.module.transformer.blocks[ll].in_x[0].mean(0).view(1, -1))

            print(rr / count)
            print(rrr / count)
            count = count + 1

        meter.add_loss_acc("ADV", {'CE': loss.item()}, (classification_result_after_attack.sum().item()), y.size(0))

        if i % 1 == 0 and False:
            logger.info("Iter: [{:d}/{:d}] Loss and Acc for all models:".format(i, int(len(data_loader) * 2)))
            msg = meter.get_loss_acc_msg()
            logger.info(msg)
            continue

        if out0.topk(1)[1] != y:
            continue
        elif out2.topk(1)[1] != y:
            #   print(torch.cuda.memory_summary())
            for batch in range(X.size(0)):
                # for l in range(12):
                if is_deit is False:

                    for l in range(layer_num):
                        aatt = model.module.transformer.blocks[l].attn.scores[0].mean(0)[1:, 1:]
                        model.module.transformer.blocks[l].get_grad = -1
                        model.module.transformer.blocks[l].attn.print = -1

                outt = model(perturb_x)

                for l in range(layer_num):
                    if is_deit:
                        break
                    model.module.transformer.blocks[l].attn.print = -1
                    model.module.transformer.blocks[l].attn.clean = -1
                    model.module.transformer.blocks[l].get_grad = -1

                if False:
                    value_list = torch.tensor([]).cuda()
                    score_list = torch.tensor([]).cuda()
                    for l in range(12):
                        if is_deit:
                            break

                        value_list = torch.cat((value_list, model.module.transformer.blocks[l].attn.proj_v(
                            model.module.transformer.blocks[l].attn.input)[0, 1:].sum(1).topk(1)[1]))
                        score_list = torch.cat((score_list,
                                                model.module.transformer.blocks[l].attn.scores[0].sum(0)[1:, 1:].sum(
                                                    0).topk(1)[1]))
                        model.module.transformer.blocks[l].attn.print = -1

                        model.module.transformer.blocks[l].attn.clean = -1

                    if i == 0 or i == 1:
                        ave_value_acc = (
                                (value_list.unique() == max_patch_index).sum() / value_list.unique().size(0)).view(
                            1, -1)
                        ave_score_acc = (
                                (score_list.unique() == max_patch_index).sum() / score_list.unique().size(0)).view(
                            1, -1)
                    else:
                        ave_value_acc = torch.cat((ave_value_acc,
                                                   ((
                                                            value_list.unique() == max_patch_index).sum() / value_list.unique().size(
                                                       0)).view(1, -1)))
                        ave_score_acc = torch.cat((ave_score_acc,
                                                   ((
                                                            score_list.unique() == max_patch_index).sum() / score_list.unique().size(
                                                       0)).view(1, -1)))
                    print('value error rate:', ave_value_acc.mean())
                    print('score error rate:', ave_score_acc.mean())

                    if i % 1 == 0 and True:
                        classification_result_after_attack = outt.max(1)[1] == y
                        loss = criterion(outt, y)
                        meter.add_loss_acc("DEF", {'CE': loss.item()},
                                           (classification_result_after_attack.sum().item()),
                                           y.size(0))
                        logger.info(
                            "Iter: [{:d}/{:d}] Loss and Acc for all models:".format(i, int(len(data_loader) * 2)))
                        msg = meter.get_loss_acc_msg()
                        logger.info(msg)
                        break

                if is_deit is False:
                    with torch.no_grad():
                        aa = nn.Softmax(dim=1)
                        out_list = torch.zeros(layer_num).cuda()
                        max_out = outt.topk(1)[1]
                        for n in range(layer_num + 1):
                            for m in range(layer_num):
                                if m == n:
                                    model.module.transformer.blocks[m].skip = True
                                else:
                                    model.module.transformer.blocks[m].skip = False
                            out22 = aa(model(perturb_x))
                            if n != layer_num:
                                if out22[0, max_out.tolist()[0]] < 0.5:
                                    layer_log[n, 1] = layer_log[n, 1] + 1
                                    if out22[0, y[0]] > 0.5:
                                        layer_log[n, 2] = layer_log[n, 2] + 1
                                out_list[n] = out22[0, max_out.tolist()[0]]
                    print('skip list of adv output', out_list)
                #  stdd = model.module.transformer.blocks[0].attn.scores[0].mean(0).mean(0)[1:].std()
                #   meann = model.module.transformer.blocks[0].attn.scores[0].mean(0).mean(0)[1:].mean()

                # save_image(   ((model.module.transformer.blocks[0].attn.scores[0].mean(0).mean(0)[1:] - meann) / stdd).view(24,24), '/home/liang/vit/inputss/a' + str(i) + '_0.png')
                #    save_image(model.module.transformer.blocks[0].attn.scores[1].mean(0).mean(0)[1:].view(24,24), '/home/liang/vit/inputss/a'+str(i)+'_1.png')
                #   save_image(model.module.transformer.blocks[0].attn.scores[2].mean(0).mean(0)[1:].view(24,24), '/home/liang/vit/inputss/a'+str(i)+'_2.png')

                #      save_image(perturb_x[1], '/home/liang/vit/inputss/' + str(i) + '_1.png')

                #     save_image(perturb_x[2], '/home/liang/vit/inputss/' + str(i) + '_2.png')
                if is_deit:
                    patch_num = int((model.module.blocks[0].attn.scores.size(2) - 1) ** 0.5)
                else:
                    patch_num = int(X.size(3) / model.module.patch_embedding.weight.size(3))
                atten_diff = torch.zeros(patch_num, patch_num).cuda()
                atten2 = torch.zeros(patch_num, patch_num).cuda()
                patch_index = []
                for l in range(layer_num):
                    if is_deit:
                        attenn = model.module.blocks[l].attn.scores[batch].mean(0).mean(0)[1:].view(patch_num,
                                                                                                    patch_num)
                        aaa, pp = model.module.blocks[l].attn.scores[0].mean(0)[1:, 1:].mean(0).topk(5)
                    else:
                        attenn = model.module.transformer.blocks[l].attn.scores[batch].mean(0).mean(0)[1:].view(
                            patch_num, patch_num)
                        aaa, pp = model.module.transformer.blocks[l].attn.scores[0].mean(0)[1:, 1:].mean(0).topk(5)
                    # atten_diff[:-1,:] += attenn[:-1,:] - attenn[1:,:]
                    #  atten_diff[1:,:] += attenn[1:,:] -  attenn[:-1,:]
                    #  atten_diff[:, 1:] += attenn[:, 1:] - attenn[:, :-1]
                    #  atten_diff[:, :-1] += attenn[:, :-1] - attenn[:, :1]
                    # print(model.module.transformer.blocks[l].attn.scores[batch].mean(0).mean(0)[1:].topk(3), model.module.transformer.blocks[l].attn.atten_grad[batch].mean(0).mean(0)[1:].topk(3))

                    if aaa[0] / aaa[1] > 1.5:
                        #  stddd = model.module.transformer.blocks[l].attn.scores[batch].mean(0)[1:,1:].std(dim = 1)
                        #    print('std:',stddd[pp[0]], stddd.mean())
                        # if stddd[pp[0]] > stddd.mean():
                        list_tau[l].append((aaa[0] / aaa[1]).detach().cpu().numpy().tolist())
                        print('found 1 gap in ', l, 'th layer, with ratio', aaa[0] / aaa[1], ' at ', pp[0].tolist())
                        patch_index.append(pp[0].tolist())
                        layer_log[l, 0] = layer_log[l, 0] + 1
                    else:
                        list_tau_2[l].append((aaa[0] / aaa[1]).detach().cpu().numpy().tolist())

                        #  if pp[0] == max_patch_index:
                    #     layer_log[l, 1] = layer_log[l, 1] + 1

                    if aaa[1] / aaa[2] > 1.5:
                        print('found 2 gap in ', l, 'th layer, with ratio', aaa[1] / aaa[2], ' at ', pp[0].tolist(),
                              pp[1].tolist())
                        patch_index.append(pp[0].tolist())
                        patch_index.append(pp[1].tolist())

                    if True:
                        if aaa[2] / aaa[3] > 1.5:
                            print('found 3 gap in ', l, 'th layer, with ratio', aaa[2] / aaa[3], ' at ', pp[0].tolist(),
                                  pp[1].tolist(), pp[2].tolist())
                            patch_index.append(pp[0].tolist())
                            patch_index.append(pp[1].tolist())
                            patch_index.append(pp[2].tolist())
                        if aaa[3] / aaa[4] > 1.4:
                            print('found 4 gap in ', l, 'th layer, with ratio', aaa[3] / aaa[4], ' at ', pp[0].tolist(),
                                  pp[1].tolist(), pp[2].tolist(), pp[3].tolist())
                            patch_index.append(pp[0].tolist())
                            patch_index.append(pp[1].tolist())
                            patch_index.append(pp[2].tolist())
                            patch_index.append(pp[3].tolist())

                    atten2 += attenn

                # aaa, pp = atten_diff.view(-1).topk(5)
                # print(aaa,pp)
                aaa, pp2 = atten2.view(-1).topk(5)
                #     print(aaa,pp)
                #   patch_index.append(pp[0].tolist())
                patch_index.append(pp2[0].tolist())
                patch_index = [*set(patch_index)]
                print(patch_index, max_patch_index)
                #   if patch_index[0] ==max_patch_index:
                #     print('success')

                perturb_x2 = perturb_x.clone()
                for aaa, p in enumerate(patch_index):
                    row_p = (p // patch_num_per_line) * patch_size
                    column_p = (p % patch_num_per_line) * patch_size

                    perturb_x[batch, 0, row_p:row_p + patch_size, column_p:column_p + patch_size] = perturb_x[
                        batch, 0].mean()
                    perturb_x[batch, 1, row_p:row_p + patch_size, column_p:column_p + patch_size] = perturb_x[
                        batch, 1].mean()
                    perturb_x[batch, 2, row_p:row_p + patch_size, column_p:column_p + patch_size] = perturb_x[
                        batch, 2].mean()

            out2 = model(perturb_x)

            '''
            save_image(perturb_x, '/home/liang/vit/123.png')
            a = []
            b = []
            for l in range(11):
                a.append(model.module.transformer.blocks[l].attn.scores.clone())
                b.append( model.module.transformer.blocks[l].attn.input )
            out = model(X)

          #  for l in range(1):
           #     model.module.transformer.blocks[l].attn.patch = max_patch_index

            out = model(perturb_x)

            for l in range(11):
                for h in range(12):
                    ratio = (model.module.transformer.blocks[l].attn.scores - a[l])[:, h].abs().sum() / 576
                    stdd = (model.module.transformer.blocks[l].attn.scores - a[l]).std()
                    if h == 0:
                        save_image((model.module.transformer.blocks[l].attn.input - b[l]), '/home/liang/vit/inputss/'+ str(l) + '.png')
                   # save_image((model.module.transformer.blocks[l].attn.scores - a[l])[:, h, 330, 1:].view(24,24) / stdd, '/home/liang/vit/atten5/'+ str(l) + '_' + str(h)+ '.png')
                   # save_image((model.module.transformer.blocks[l].attn.scores - a[l])[:, h]/stdd, '/home/liang/vit/atten2/' + str(l) + '_' + str(h) + '_' + str( ratio.item()) + '.png')
            '''

            classification_result_after_defense = out2.max(1)[1] == y
            if classification_result_after_defense != True:
                i = i
            loss2 = criterion(out2, y)
            meter.add_loss_acc("DEF1", {'CE': loss2.item()}, (classification_result_after_defense.sum().item()),
                               y.size(0))

        '''Message'''
        if i % 1 == 0:
            logger.info("Iter: [{:d}/{:d}] Loss and Acc for all models:".format(i, int(len(data_loader) * 2)))
            msg = meter.get_loss_acc_msg()
            logger.info(msg)
            print(layer_log.transpose(1, 0))
            f=open('aa.txt','a')
            f.write(str(list_tau))
            f.close()
            f=open('bb.txt','a')
            f.write(str(list_tau_2))
            f.close()

        #   classification_result_after_attack = classification_result_after_attack[classification_result == False]
        #    false2true_num += classification_result_after_attack.sum().item()
        #    logger.info("Total False -> True: {}".format(false2true_num))

    end_time = time.time()
    msg = meter.get_loss_acc_msg()
    logger.info("\n Finish! Using time: {}\n{}".format((end_time - start_time), msg))


if __name__ == "__main__":
    main()
