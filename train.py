import numpy as np
import torch.optim as optim
from torch import nn
from tqdm import tqdm
import itertools

from conf import *
from models.decoder import Decoder
from models.encoder import Encoder

def EMA(target_model, online_model,mu):
    for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):  #mu 逐渐增大
        target_param.data = target_param.data.clone() * mu + online_param.data.clone() * (1-mu)

def train_stage1(dataloader, attribute_dims, max_len, b1=0.5, b2=0.999):
    '''
    :param dataloader:
    :param attribute_dims:  Number of attribute values per attribute : list
    :param max_len:  max length of traces
    :param b1: adam: decay of first order momentum of gradient
    :param b2: adam: decay of first order momentum of gradient
    :return:
    '''

    teacher_encoder =  Encoder(attribute_dims, max_len, d_model, ffn_hidden, n_heads, n_layers, n_layers_agg, drop_prob, device)
    for param in teacher_encoder.parameters(): # for self-distillation （no gradient）
        param.requires_grad = False
    encoder = Encoder(attribute_dims, max_len, d_model, ffn_hidden, n_heads, n_layers, n_layers_agg, drop_prob, device)
    decoder = Decoder(attribute_dims, max_len, d_model, ffn_hidden, n_heads, n_layers, drop_prob, device)

    teacher_encoder.to(device)
    encoder.to(device)
    decoder.to(device)

    mseLoss=nn.MSELoss()


    optimizer = torch.optim.Adam(itertools.chain(encoder.parameters(),decoder.parameters()),lr=lr_stage1, betas=(b1, b2))

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(epoch_stage1/2), gamma=0.1)

    EMA(teacher_encoder, encoder, 0)  #使 teacher_encoder 和 encoder 初始化参数相同

    if mu_start ==  mu_end:
        mus = [mu_start for _ in range(epoch_stage1)]
    else:
        mus = np.arange(mu_start, mu_end, (mu_end - mu_start) / epoch_stage1)

    print("*"*10+"training-stage1"+"*"*10)
    for epoch in range(int(epoch_stage1)):
        train_loss = 0.0
        train_num = 0
        mu=mus[epoch]
        for i, Xs in enumerate(tqdm(dataloader)):
            EMA(teacher_encoder,encoder,mu)
            mask = Xs[-1]  #sequence mask
            Xs = Xs[:-1]
            masked_Xs=[]
            masks=[]   #mask for training mask autoencoder
            for k ,X in enumerate(Xs):
                ##mask the input (mask some attribute values)
                Xs[k] = X.to(device)
                shape = X.shape
                X = X.flatten()
                masked_index = np.random.choice(list(range(len(X))),
                                                size=int(len(X) * masked_ratio), replace=False)
                unmasked_index=list(set(range(len(X)))-set(masked_index))
                cp_mask=mask.clone().flatten()
                cp_mask[unmasked_index]=False  # 只计算那些被mask的属性值， False不计算loss
                cp_mask=cp_mask.reshape(shape)
                masks.append(cp_mask.to(device))
                X[masked_index] = 0
                X=X.reshape(shape)
                masked_Xs.append(X.to(device))
            mask=mask.to(device)


            optimizer.zero_grad()

            target_enc_output = teacher_encoder(Xs, mask)
            enc_output = encoder(masked_Xs, mask)
            fake_X = decoder(Xs, enc_output, mask)

            reconstruction_loss = 0.0
            concat_mask = torch.cat(masks, dim=1)
            hidden_loss = mseLoss(target_enc_output[concat_mask],enc_output[concat_mask])

            for ij in range(len(attribute_dims)):
                # --------------
                # 除了每一个属性的起始字符之外,其他重建误差
                # ---------------
                pred = torch.softmax(fake_X[ij][:, :-1, :], dim=2).flatten(0, -2) #最后一个预测无意义
                true = Xs[ij][:, 1:].flatten()

                corr_pred = pred.gather(1, true.view(-1, 1)).flatten().to(device).reshape(-1,
                                                                                          fake_X[0].shape[1] - 1)

                cross_entropys = -torch.log(corr_pred)
                reconstruction_loss += cross_entropys.masked_select((masks[ij][:, 1:])).mean() #只算了那些被mask的值
            loss = reconstruction_loss + beta * hidden_loss
            train_loss += loss.item() * Xs[0].shape[0]
            train_num +=Xs[0].shape[0]
            loss.backward()
            optimizer.step()
        ## 计算一个epoch在训练集上的损失和精度
        train_loss_epoch=train_loss / train_num
        print(f"[Epoch {epoch+1:{len(str(epoch_stage1))}}/{epoch_stage1}] "
                f"[lr {optimizer.param_groups[0]['lr']}] "
              f"[loss: {train_loss_epoch:3f}]")
        scheduler.step()

    return encoder,decoder


def train_stage2(dataloader,encoder,decoder, attribute_dims, b1=0.5, b2=0.999):
    '''
    :param dataloader:
    :param encoder:
    :param decoder:
    :param attribute_dims:  Number of attribute values per attribute : list
    :param b1: adam: decay of first order momentum of gradient
    :param b2: adam: decay of first order momentum of gradient
    :return:
    '''

    encoder.to(device)
    decoder.to(device)

    ###设置encoder不参与训练，不进行梯度下降（固定encoder的梯度）
    for param in encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(decoder.parameters(),lr=lr_stage2, betas=(b1, b2))

    scheduler =  torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    print("*"*10+"training-stage2"+"*"*10)
    for epoch in range(int(epoch_stage2)):
        train_loss = 0.0
        train_num = 0
        for i, Xs in enumerate(tqdm(dataloader)):
            mask = Xs[-1]
            Xs = Xs[:-1]
            mask = mask.to(device)
            for k, X in enumerate(Xs):
                Xs[k] = X.to(device)

            optimizer.zero_grad()

            enc_output = encoder(Xs, mask)
            fake_X = decoder(Xs, enc_output, mask)

            loss = 0.0
            for ij in range(len(attribute_dims)):
                # --------------
                # 除了每一个属性的起始字符之外,其他重建误差
                # ---------------
                pred = torch.softmax(fake_X[ij][:, :-1, :], dim=2).flatten(0, -2)  # 最后一个预测无意义
                true = Xs[ij][:, 1:].flatten()

                corr_pred = pred.gather(1, true.view(-1, 1)).flatten().to(device).reshape(-1,
                                                                                          fake_X[0].shape[1] - 1)

                cross_entropys = -torch.log(corr_pred)
                loss += cross_entropys.masked_select((mask[:, 1:])).mean()

            train_loss += loss.item() * Xs[0].shape[0]
            train_num += Xs[0].shape[0]
            loss.backward()
            optimizer.step()
        ## 计算一个epoch在训练集上的损失和精度
        train_loss_epoch = train_loss / train_num
        print(f"[Epoch {epoch + 1:{len(str(epoch_stage2))}}/{epoch_stage2}] "
              f"[lr {optimizer.param_groups[0]['lr']}] "
              f"[loss: {train_loss_epoch:3f}]")
        scheduler.step()

    return encoder,decoder

