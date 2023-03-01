from util.log import Logger
from util.get_mask import get_mask,get_element_mask
from model.model import MECOM
import torch.nn as nn
import random
import torch.optim as optim
import torch
import argparse
import numpy as np
import os

def label_processing(label_train, device):
    label_train = torch.squeeze(label_train)
    label_train = label_train.type(torch.int64).to(device)
    return label_train

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def standardization(a):
    mean_a = torch.mean(a,dim=1)
    std_a = torch.std(a,dim=1)
    n_a = a.sub_(mean_a[:, None]).div_(std_a[:, None])
    return n_a

def standardization_9(a):
    count = a.size()[0]
    for i in range(count):
        a[i] = standardization(a[i])
    return a

def Loss_re1(res,gt,sn):
    return torch.sum(torch.sum(torch.pow(res-gt,2),dim=1) * sn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--missing_rate', type=float, default=0.5)

    parser.add_argument('--gpu', type=str, default='2', metavar='N')
    parser.add_argument('--log', type=str, default='log_2', metavar='N')
    parser.add_argument('--model_pth', type=str, default='model_2.pth', metavar='N')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--meta_tasks_epochs', type=int, default=1, metavar='N')
    parser.add_argument('--unified_epoch', type=int, default=2, metavar='N')

    parser.add_argument('--drop_pro', type=float, default=0.1, metavar='N')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='N')
    parser.add_argument('--lr_unified', type=float, default=0.0001, metavar='N')

    args = parser.parse_args()

    # gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # seed
    seed_everything(1)
    #log
    logger = Logger(args.log)
    logger.info("missing_rate = {:.1f} ===> dropout = {:.1f} ===>  ".format(args.missing_rate, args.drop_pro))

    model = MECOM(args.drop_pro)
    # torch.save(model.state_dict(), args.model_pth)

    # load data
    if 1:
        # feature_image_train_200 = torch.load('path')
        # feature_image_test_200 = torch.load('path')
        # feature_audio_train_200 = torch.load('path')
        # feature_audio_test_200 = torch.load('path')
        # feature_text_train_200 = torch.load('path')
        # feature_text_test_200 = torch.load('path')
        # feature_video_train_200 = torch.load('path')
        # feature_video_test_200 = torch.load('path')
        #
        # feature_image_train_200_9 = torch.load('path')
        # feature_image_test_200_9 = torch.load('path')
        # feature_audio_train_200_9 = torch.load('path')
        # feature_audio_test_200_9 = torch.load('path')
        # feature_text_train_200_9 = torch.load('path')
        # feature_text_test_200_9 = torch.load('path')
        # feature_video_train_200_9 = torch.load('path.pth')
        # feature_video_test_200_9 = torch.load('path.pth')
        #
        # label_train_200 = torch.load('path')
        # label_test_200 = torch.load('path')
        #
        # tr200_len = feature_image_train_200.size()[0]
        # te200_len = feature_image_test_200.size()[0]
        feature_image_train_200 = torch.zeros(size=(2994, 2048))
        feature_image_test_200 = torch.zeros(size=(2930, 2048))
        feature_audio_train_200 = torch.zeros(size=(2994, 2048))
        feature_audio_test_200 = torch.zeros(size=(2930, 2048))
        feature_text_train_200 = torch.zeros(size=(2994, 2048))
        feature_text_test_200 = torch.zeros(size=(2930, 2048))
        feature_video_train_200 = torch.zeros(size=(2994, 2048))
        feature_video_test_200 = torch.zeros(size=(2930, 2048))
        length = 16
        feature_image_train_200_9 = torch.zeros(size=(2994, length, 2048))
        feature_image_test_200_9 = torch.zeros(size=(2930, length, 2048))
        feature_audio_train_200_9 = torch.zeros(size=(2994, length, 2048))
        feature_audio_test_200_9 = torch.zeros(size=(2930, length, 2048))
        feature_text_train_200_9 = torch.zeros(size=(2994, 17, 2048))
        feature_text_test_200_9 = torch.zeros(size=(2930, 17, 2048))
        feature_video_train_200_9 = torch.zeros(size=(2994, length, 2048))
        feature_video_test_200_9 = torch.zeros(size=(2930, length, 2048))

        label_train_200 = torch.zeros(size=(2994, 1))
        label_test_200 = torch.zeros(size=(2930, 1))

        tr200_len = feature_image_train_200.size()[0]
        te200_len = feature_image_test_200.size()[0]
    # to device and standardization
    if 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        label_train_200 = label_processing(label_train_200, device)
        label_test_200 = label_processing(label_test_200, device)

        feature_image_train_200 = standardization(feature_image_train_200)
        feature_audio_train_200 = standardization(feature_audio_train_200)
        feature_text_train_200 = standardization(feature_text_train_200)
        feature_video_train_200 = standardization(feature_video_train_200)

        feature_image_test_200 = standardization(feature_image_test_200)
        feature_audio_test_200 = standardization(feature_audio_test_200)
        feature_text_test_200 = standardization(feature_text_test_200)
        feature_video_test_200 = standardization(feature_video_test_200)

        feature_image_train_200_9 = standardization_9(feature_image_train_200_9)
        feature_audio_train_200_9 = standardization_9(feature_audio_train_200_9)
        feature_text_train_200_9 = standardization_9(feature_text_train_200_9)
        feature_video_train_200_9 = standardization_9(feature_video_train_200_9)

        feature_image_test_200_9 = standardization_9(feature_image_test_200_9)
        feature_audio_test_200_9 = standardization_9(feature_audio_test_200_9)
        feature_text_test_200_9 = standardization_9(feature_text_test_200_9)
        feature_video_test_200_9 = standardization_9(feature_video_test_200_9)

    test_acc_all = 0
    test_acc_all_1 = 0
    for zz in range(5):
        # generate mask data
        if 1:
            mask_train_200 = get_mask(4, tr200_len, 0)
            mask_test_200 = get_mask(4, te200_len, 0)

            mask_train_200_e = get_element_mask(4, tr200_len, args.missing_rate)
            mask_test_200_e = get_element_mask(4, te200_len, args.missing_rate)

            feature_image_train_200_mask_9 = feature_image_train_200_9 * mask_train_200_e[0].unsqueeze(dim=1)
            feature_audio_train_200_mask_9 = feature_audio_train_200_9 * mask_train_200_e[1].unsqueeze(dim=1)
            feature_text_train_200_mask_9 = feature_text_train_200_9 * mask_train_200_e[2].unsqueeze(dim=1)
            feature_video_train_200_mask_9 = feature_video_train_200_9 * mask_train_200_e[3].unsqueeze(dim=1)

            feature_image_test_200_mask_9 = feature_image_test_200_9 * mask_test_200_e[0].unsqueeze(dim=1)
            feature_audio_test_200_mask_9 = feature_audio_test_200_9 * mask_test_200_e[1].unsqueeze(dim=1)
            feature_text_test_200_mask_9 = feature_text_test_200_9 * mask_test_200_e[2].unsqueeze(dim=1)
            feature_video_test_200_mask_9 = feature_video_test_200_9 * mask_test_200_e[3].unsqueeze(dim=1)

            feature_image_train_200_mask = feature_image_train_200 * mask_train_200_e[0]
            feature_audio_train_200_mask = feature_audio_train_200 * mask_train_200_e[1]
            feature_text_train_200_mask = feature_text_train_200 * mask_train_200_e[2]
            feature_video_train_200_mask = feature_video_train_200 * mask_train_200_e[3]

            feature_image_test_200_mask = feature_image_test_200 * mask_test_200_e[0]
            feature_audio_test_200_mask = feature_audio_test_200 * mask_test_200_e[1]
            feature_text_test_200_mask = feature_text_test_200 * mask_test_200_e[2]
            feature_video_test_200_mask = feature_video_test_200 * mask_test_200_e[3]

            feature_image_train_200_mask = feature_image_train_200_mask.to(device)
            feature_audio_train_200_mask = feature_audio_train_200_mask.to(device)
            feature_text_train_200_mask = feature_text_train_200_mask.to(device)
            feature_video_train_200_mask = feature_video_train_200_mask.to(device)

            feature_image_test_200_mask = feature_image_test_200_mask.to(device)
            feature_audio_test_200_mask = feature_audio_test_200_mask.to(device)
            feature_text_test_200_mask = feature_text_test_200_mask.to(device)
            feature_video_test_200_mask = feature_video_test_200_mask.to(device)

            feature_image_train_200_mask_9 = feature_image_train_200_mask_9.to(device)
            feature_audio_train_200_mask_9 = feature_audio_train_200_mask_9.to(device)
            feature_text_train_200_mask_9 = feature_text_train_200_mask_9.to(device)
            feature_video_train_200_mask_9 = feature_video_train_200_mask_9.to(device)

            feature_image_test_200_mask_9 = feature_image_test_200_mask_9.to(device)
            feature_audio_test_200_mask_9 = feature_audio_test_200_mask_9.to(device)
            feature_text_test_200_mask_9 = feature_text_test_200_mask_9.to(device)
            feature_video_test_200_mask_9 = feature_video_test_200_mask_9.to(device)

            mask_train_200 = mask_train_200.to(device)
            mask_test_200 = mask_test_200.to(device)
            mask_train_200_re = (mask_train_200 - 1) * (-1)
            mask_test_200_re = (mask_test_200 - 1) * (-1)

        # initialize model
        # model.load_state_dict(torch.load(args.model_pth))

        loss_function = nn.CrossEntropyLoss().to(device)
        optimizer_t = optim.Adam(model.parameters(), lr=args.lr)
        b = args.batch_size
        train_round = int(tr200_len / b)
        test_round = int(te200_len / b)
        train_yu = int(tr200_len % b)
        best_acc = 0
        best_acc_1 = 0
        best_acc_9 = 0

        for epoch in range(args.meta_tasks_epochs):
            model.train()
            optimizer_t.zero_grad()
            for i in range(train_round + 1):
                if (i == train_round):
                    logits, image_all, audio_all, text_all, video_all, deco_image, deco_audio, deco_text, deco_video, x,\
                    image_all_9, audio_all_9, text_all_9, video_all_9, deco_image_9, deco_audio_9, deco_text_9, deco_video_9, x_9, logits_1, logits_9,final_x = model(
                        feature_image_train_200_mask[i * b:2994], feature_audio_train_200_mask[i * b:2994], feature_text_train_200_mask[i * b:2994],
                        feature_video_train_200_mask[i * b:2994],
                        feature_image_train_200_mask_9[i * b:2994], feature_audio_train_200_mask_9[i * b:2994], feature_text_train_200_mask_9[i * b:2994],
                        feature_video_train_200_mask_9[i * b:2994], mask_train_200[:, i * b:2994], mask_train_200_re[:, i * b:2994])

                    loss_cls = loss_function(logits, label_train_200[i * b:2994])+loss_function(logits_1, label_train_200[i * b:2994])+loss_function(logits_9, label_train_200[i * b:2994])

                    loss_reconstuction = (Loss_re1(deco_image, image_all, mask_train_200[0, i * b:2994]) + Loss_re1(deco_audio, audio_all, mask_train_200[1, i * b:2994]) + \
                                          Loss_re1(deco_text, text_all, mask_train_200[2, i * b:2994]) + Loss_re1(deco_video, video_all, mask_train_200[3, i * b:2994])+ \
                                          Loss_re1(deco_image_9, image_all_9, mask_train_200[0, i * b:2994]) + Loss_re1(deco_audio_9, audio_all_9,mask_train_200[1, i * b:2994])+ \
                                          Loss_re1(deco_text_9, text_all_9, mask_train_200[2, i * b:2994]) + Loss_re1(deco_video_9, video_all_9, mask_train_200[3, i * b:2994])
                                          ) / 2048 / train_yu
                    loss = loss_cls + loss_reconstuction

                    optimizer_t.zero_grad()
                    loss.backward()
                    optimizer_t.step()
                else:
                    logits, image_all, audio_all, text_all, video_all, deco_image, deco_audio, deco_text, deco_video, x, \
                    image_all_9, audio_all_9, text_all_9, video_all_9, deco_image_9, deco_audio_9, deco_text_9, deco_video_9, x_9, logits_1, logits_9,final_x = model(
                        feature_image_train_200_mask[i * b:(i + 1) * b], feature_audio_train_200_mask[i * b:(i + 1) * b], feature_text_train_200_mask[i * b:(i + 1) * b],
                        feature_video_train_200_mask[i * b:(i + 1) * b],
                        feature_image_train_200_mask_9[i * b:(i + 1) * b], feature_audio_train_200_mask_9[i * b:(i + 1) * b], feature_text_train_200_mask_9[i * b:(i + 1) * b],
                        feature_video_train_200_mask_9[i * b:(i + 1) * b],mask_train_200[:, i * b:(i + 1) * b], mask_train_200_re[:, i * b:(i + 1) * b])

                    loss_cls = loss_function(logits, label_train_200[i * b:(i + 1) * b])+loss_function(logits_1, label_train_200[i * b:(i + 1) * b])+loss_function(logits_9, label_train_200[i * b:(i + 1) * b])
                    loss_reconstuction = (Loss_re1(deco_image, image_all, mask_train_200[0, i * b:(i + 1) * b]) + Loss_re1(deco_audio, audio_all, mask_train_200[1, i * b:(i + 1) * b]) + \
                                          Loss_re1(deco_text, text_all, mask_train_200[2, i * b:(i + 1) * b]) + Loss_re1(deco_video, video_all,mask_train_200[3, i * b:(i + 1) * b])+ \
                                          Loss_re1(deco_image_9, image_all_9, mask_train_200[0, i * b:(i + 1) * b]) + Loss_re1(deco_audio_9, audio_all_9,mask_train_200[1, i * b:(i + 1) * b]) + \
                                          Loss_re1(deco_text_9, text_all_9, mask_train_200[2, i * b:(i + 1) * b]) + Loss_re1(deco_video_9, video_all_9,mask_train_200[3, i * b:(i + 1) * b])
                                          ) / 2048 / b
                    loss = loss_cls + loss_reconstuction
                    optimizer_t.zero_grad()
                    loss.backward()
                    optimizer_t.step()

            model.eval()
            value_acc = 0
            value_acc_1 = 0
            value_acc_9 = 0
            test_round = int(2930 / b)
            for i in range(test_round + 1):
                if (i == test_round):
                    with torch.no_grad():
                        logits, image_all, audio_all, text_all, video_all, deco_image, deco_audio, deco_text, deco_video, x, \
                        image_all_9, audio_all_9, text_all_9, video_all_9, deco_image_9, deco_audio_9, deco_text_9, deco_video_9, x_9, logits_1, logits_9,final_x = model(
                            feature_image_test_200_mask[i * b:2930], feature_audio_test_200_mask[i * b:2930], feature_text_test_200_mask[i * b:2930],
                            feature_video_test_200_mask[i * b:2930],
                            feature_image_test_200_mask_9[i * b:2930], feature_audio_test_200_mask_9[i * b:2930], feature_text_train_200_mask_9[i * b:2930],
                            feature_video_test_200_mask_9[i * b:2930],mask_test_200[:, i * b:2930], mask_test_200_re[:, i * b:2930])
                        test_predict = torch.max(logits, dim=1)[1]
                        value_acc += torch.eq(test_predict, label_test_200[i * b:2930]).sum().item()
                        test_predict = torch.max(logits_1, dim=1)[1]
                        value_acc_1 += torch.eq(test_predict, label_test_200[i * b:2930]).sum().item()
                        test_predict = torch.max(logits_9, dim=1)[1]
                        value_acc_9 += torch.eq(test_predict, label_test_200[i * b:2930]).sum().item()
                else:
                    with torch.no_grad():
                        logits, image_all, audio_all, text_all, video_all, deco_image, deco_audio, deco_text, deco_video, x, \
                        image_all_9, audio_all_9, text_all_9, video_all_9, deco_image_9, deco_audio_9, deco_text_9, deco_video_9, x_9, logits_1, logits_9,final_x = model(
                            feature_image_test_200_mask[i * b:(i + 1) * b], feature_audio_test_200_mask[i * b:(i + 1) * b], feature_text_test_200_mask[i * b:(i + 1) * b],
                            feature_video_test_200_mask[i * b:(i + 1) * b],
                            feature_image_test_200_mask_9[i * b:(i + 1) * b], feature_audio_test_200_mask_9[i * b:(i + 1) * b], feature_text_test_200_mask_9[i * b:(i + 1) * b],
                            feature_video_test_200_mask_9[i * b:(i + 1) * b],mask_test_200[:, i * b:(i + 1) * b], mask_test_200_re[:, i * b:(i + 1) * b])
                        test_predict = torch.max(logits, dim=1)[1]
                        value_acc += torch.eq(test_predict, label_test_200[i * b:(i + 1) * b]).sum().item()
                        test_predict = torch.max(logits_1, dim=1)[1]
                        value_acc_1 += torch.eq(test_predict, label_test_200[i * b:(i + 1) * b]).sum().item()
                        test_predict = torch.max(logits_9, dim=1)[1]
                        value_acc_9 += torch.eq(test_predict, label_test_200[i * b:(i + 1) * b]).sum().item()
            value_acc = value_acc / 2930
            value_acc_1 = value_acc_1 / 2930
            value_acc_9 = value_acc_9 / 2930
            if (value_acc_1 > best_acc_1):
                best_acc_1 = value_acc_1
            if (value_acc_9 > best_acc_9):
                best_acc_9 = value_acc_9
            if (value_acc > best_acc):
                best_epoch = epoch
                best_acc = value_acc

            logger.info("best_epoch = {:.0f} ===> epoch = {:.0f} ===> loss = {:.6f} ===> best_acc = {:.4f} ===> acc = {:.4f} ===> best_acc_9 = {:.4f} ===> acc_9 = {:.4f} ===> best_acc_all = {:.4f} ===> acc_all = {:.4f}"
                                .format(best_epoch + 1, epoch + 1, loss, best_acc_1, value_acc_1,best_acc_9,value_acc_9,best_acc,value_acc))

        # unified
        model.eval()
        aa=torch.zeros(size=(2994,2048))
        aa_9 = torch.zeros(size=(2994, 2048))
        aa=aa.to(device)
        aa_9 = aa_9.to(device)
        for i in range(train_round + 1):
            if (i == train_round):
                with torch.no_grad():
                    logits, image_all, audio_all, text_all, video_all, deco_image, deco_audio, deco_text, deco_video, x, \
                    image_all_9, audio_all_9, text_all_9, video_all_9, deco_image_9, deco_audio_9, deco_text_9, deco_video_9, x_9, logits_1, logits_9, final_x = model(
                        feature_image_train_200_mask[i * b:2994], feature_audio_train_200_mask[i * b:2994], feature_text_train_200_mask[i * b:2994],
                        feature_video_train_200_mask[i * b:2994],
                        feature_image_train_200_mask_9[i * b:2994], feature_audio_train_200_mask_9[i * b:2994], feature_text_train_200_mask_9[i * b:2994],
                        feature_video_train_200_mask_9[i * b:2994], mask_train_200[:, i * b:2994], mask_train_200_re[:, i * b:2994])
                    aa[i * b:2994,:]=x
                    aa_9[i * b:2994, :] = x_9
            else:
                with torch.no_grad():
                    logits, image_all, audio_all, text_all, video_all, deco_image, deco_audio, deco_text, deco_video, x, \
                    image_all_9, audio_all_9, text_all_9, video_all_9, deco_image_9, deco_audio_9, deco_text_9, deco_video_9, x_9, logits_1, logits_9, final_x = model(
                        feature_image_train_200_mask[i * b:(i + 1) * b], feature_audio_train_200_mask[i * b:(i + 1) * b], feature_text_train_200_mask[i * b:(i + 1) * b],
                        feature_video_train_200_mask[i * b:(i + 1) * b],
                        feature_image_train_200_mask_9[i * b:(i + 1) * b], feature_audio_train_200_mask_9[i * b:(i + 1) * b], feature_text_train_200_mask_9[i * b:(i + 1) * b],
                        feature_video_train_200_mask_9[i * b:(i + 1) * b], mask_train_200[:, i * b:(i + 1) * b], mask_train_200_re[:, i * b:(i + 1) * b])
                    aa[i * b:(i + 1) * b,:] = x
                    aa_9[i * b:(i + 1) * b,:] = x_9

        best_acc = 0
        best_acc_1 = 0
        best_acc_9 = 0
        optimizer_t = optim.Adam(model.parameters(), lr=args.lr_unified)
        for epoch in range(args.unified_epoch):
            model.train()
            optimizer_t.zero_grad()
            for i in range(train_round + 1):
                if (i == train_round):
                    logits, image_all, audio_all, text_all, video_all, deco_image, deco_audio, deco_text, deco_video, x, \
                    image_all_9, audio_all_9, text_all_9, video_all_9, deco_image_9, deco_audio_9, deco_text_9, deco_video_9, x_9, logits_1, logits_9,final_x = model(
                        feature_image_train_200_mask[i * b:2994], feature_audio_train_200_mask[i * b:2994], feature_text_train_200_mask[i * b:2994],
                        feature_video_train_200_mask[i * b:2994],
                        feature_image_train_200_mask_9[i * b:2994], feature_audio_train_200_mask_9[i * b:2994], feature_text_train_200_mask_9[i * b:2994],
                        feature_video_train_200_mask_9[i * b:2994], mask_train_200[:, i * b:2994], mask_train_200_re[:, i * b:2994])

                    loss_latent_re = (Loss_re1(image_all, aa[i * b:2994], mask_train_200[0, i * b:2994]) + Loss_re1(audio_all, aa[i * b:2994], mask_train_200[1, i * b:2994]) + \
                                      Loss_re1(text_all, aa[i * b:2994], mask_train_200[2, i * b:2994]) + Loss_re1(video_all, aa[i * b:2994], mask_train_200[3, i * b:2994]) + \
                                      Loss_re1(image_all_9, aa_9[i * b:2994], mask_train_200[0, i * b:2994]) + Loss_re1(audio_all_9, aa_9[i * b:2994],mask_train_200[1, i * b:2994]) + \
                                      Loss_re1(text_all_9, aa_9[i * b:2994], mask_train_200[2, i * b:2994]) + Loss_re1(video_all_9, aa_9[i * b:2994],mask_train_200[3, i * b:2994])
                                      ) / 2048 / train_yu

                    loss = loss_latent_re
                    optimizer_t.zero_grad()
                    loss.backward()
                    optimizer_t.step()

                else:
                    logits, image_all, audio_all, text_all, video_all, deco_image, deco_audio, deco_text, deco_video, x, \
                    image_all_9, audio_all_9, text_all_9, video_all_9, deco_image_9, deco_audio_9, deco_text_9, deco_video_9, x_9, logits_1, logits_9,final_x = model(
                        feature_image_train_200_mask[i * b:(i + 1) * b], feature_audio_train_200_mask[i * b:(i + 1) * b], feature_text_train_200_mask[i * b:(i + 1) * b],
                        feature_video_train_200_mask[i * b:(i + 1) * b],
                        feature_image_train_200_mask_9[i * b:(i + 1) * b], feature_audio_train_200_mask_9[i * b:(i + 1) * b], feature_text_train_200_mask_9[i * b:(i + 1) * b],
                        feature_video_train_200_mask_9[i * b:(i + 1) * b], mask_train_200[:, i * b:(i + 1) * b], mask_train_200_re[:, i * b:(i + 1) * b])

                    loss_latent_re = (Loss_re1(image_all, aa[i * b:(i + 1) * b], mask_train_200[0, i * b:(i + 1) * b]) + Loss_re1(audio_all, aa[i * b:(i + 1) * b], mask_train_200[1,  i * b:(i + 1) * b]) + \
                                      Loss_re1(text_all, aa[i * b:(i + 1) * b], mask_train_200[2,  i * b:(i + 1) * b]) + Loss_re1(video_all, aa[i * b:(i + 1) * b], mask_train_200[3,  i * b:(i + 1) * b]) + \
                                      Loss_re1(image_all_9, aa_9[i * b:(i + 1) * b], mask_train_200[0,  i * b:(i + 1) * b]) + Loss_re1(audio_all_9, aa_9[i * b:(i + 1) * b], mask_train_200[1,  i * b:(i + 1) * b]) + \
                                      Loss_re1(text_all_9, aa_9[i * b:(i + 1) * b], mask_train_200[2,  i * b:(i + 1) * b]) + Loss_re1(video_all_9, aa_9[i * b:(i + 1) * b], mask_train_200[3,  i * b:(i + 1) * b])
                                      ) / 2048 / b

                    loss = loss_latent_re
                    optimizer_t.zero_grad()
                    loss.backward()
                    optimizer_t.step()

            # validation
            model.eval()
            value_acc = 0
            value_acc_1 = 0
            value_acc_9 = 0
            for i in range(test_round + 1):
                if (i == test_round):
                    with torch.no_grad():
                        logits, image_all, audio_all, text_all, video_all, deco_image, deco_audio, deco_text, deco_video, x, \
                        image_all_9, audio_all_9, text_all_9, video_all_9, deco_image_9, deco_audio_9, deco_text_9, deco_video_9, x_9, logits_1, logits_9,final_x = model(
                            feature_image_test_200_mask[i * b:2930], feature_audio_test_200_mask[i * b:2930], feature_text_test_200_mask[i * b:2930],
                            feature_video_test_200_mask[i * b:2930],
                            feature_image_test_200_mask_9[i * b:2930], feature_audio_test_200_mask_9[i * b:2930], feature_text_train_200_mask_9[i * b:2930],
                            feature_video_test_200_mask_9[i * b:2930], mask_test_200[:, i * b:2930], mask_test_200_re[:, i * b:2930])
                        test_predict = torch.max(logits, dim=1)[1]
                        value_acc += torch.eq(test_predict, label_test_200[i * b:2930]).sum().item()
                        test_predict = torch.max(logits_1, dim=1)[1]
                        value_acc_1 += torch.eq(test_predict, label_test_200[i * b:2930]).sum().item()
                        test_predict = torch.max(logits_9, dim=1)[1]
                        value_acc_9 += torch.eq(test_predict, label_test_200[i * b:2930]).sum().item()
                else:
                    with torch.no_grad():
                        logits, image_all, audio_all, text_all, video_all, deco_image, deco_audio, deco_text, deco_video, x, \
                        image_all_9, audio_all_9, text_all_9, video_all_9, deco_image_9, deco_audio_9, deco_text_9, deco_video_9, x_9, logits_1, logits_9,final_x = model(
                            feature_image_test_200_mask[i * b:(i + 1) * b], feature_audio_test_200_mask[i * b:(i + 1) * b], feature_text_test_200_mask[i * b:(i + 1) * b],
                            feature_video_test_200_mask[i * b:(i + 1) * b],
                            feature_image_test_200_mask_9[i * b:(i + 1) * b], feature_audio_test_200_mask_9[i * b:(i + 1) * b], feature_text_test_200_mask_9[i * b:(i + 1) * b],
                            feature_video_test_200_mask_9[i * b:(i + 1) * b], mask_test_200[:, i * b:(i + 1) * b], mask_test_200_re[:, i * b:(i + 1) * b])
                        test_predict = torch.max(logits, dim=1)[1]
                        value_acc += torch.eq(test_predict, label_test_200[i * b:(i + 1) * b]).sum().item()
                        test_predict = torch.max(logits_1, dim=1)[1]
                        value_acc_1 += torch.eq(test_predict, label_test_200[i * b:(i + 1) * b]).sum().item()
                        test_predict = torch.max(logits_9, dim=1)[1]
                        value_acc_9 += torch.eq(test_predict, label_test_200[i * b:(i + 1) * b]).sum().item()

            value_acc = value_acc / 2930
            value_acc_1 = value_acc_1 / 2930
            value_acc_9 = value_acc_9 / 2930
            if (value_acc_1 > best_acc_1):
                best_acc_1 = value_acc_1
            if (value_acc_9 > best_acc_9):
                best_acc_9 = value_acc_9
            if (value_acc > best_acc):
                best_epoch = epoch
                best_acc = value_acc

            logger.info(
                "best_epoch = {:.0f} ===> epoch = {:.0f} ===> loss = {:.6f} ===> best_acc = {:.4f} ===> acc = {:.4f} ===> best_acc_9 = {:.4f} ===> acc_9 = {:.4f} ===> best_acc_all = {:.4f} ===> acc_all = {:.4f}"
                .format(best_epoch + 1, epoch + 1, loss, best_acc_1, value_acc_1, best_acc_9, value_acc_9, best_acc, value_acc))
        test_acc_all+=best_acc

    print(test_acc_all / 5)




