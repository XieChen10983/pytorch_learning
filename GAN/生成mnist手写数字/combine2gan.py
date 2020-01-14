# coding=gbk
from gen_random_noise import gen_noise
import numpy as np
# from discriminator_loss import discriminator_loss


def gan(Train_loader, DISCRIMINATOR, GENERATOR, D_optimizer, G_optimizer, D_loss, G_loss, show_every=250,
        Batch_size=128, Noise_feature=96, Num_epochs=20):
    Iter_count = 0
    image_set = []
    for Epoch in range(Num_epochs):
        print('----------------------------------------EPOCH', Epoch, '-------------------------------------------')
        for Batch_index, (Data, _) in enumerate(Train_loader):
            print('----------------------------Epoch', Epoch+1, '__Batch', Batch_index, '----------------------------')
            if len(Data) != Batch_size:
                continue

            # -----------------------------���濪ʼѵ���б���------------------------------
            D_optimizer.zero_grad()
            Real_images = Data.float()           # ��ͼ�������ת��Ϊfloat��ʽ�����СΪ0-1
            Real_images = 2 * (Real_images - 0.5)  # ��ͼ������ش�Сת��Ϊ-1 ~ 1
            Real_logits = DISCRIMINATOR(Real_images).float()  # ������Ϊ��ʵ��ͼ������

            G_seed_noise = gen_noise(Batch_size, Noise_feature).float()
            Fake_images = GENERATOR(G_seed_noise).detach()
            Fake_logits = DISCRIMINATOR(Fake_images.view(Batch_size, 1, 28, 28))
            Discriminator_total_loss = D_loss(Real_logits, Fake_logits)
            # print('D_loss:', Discriminator_total_loss.detach().numpy())
            Discriminator_total_loss.backward()
            D_optimizer.step()

            # -----------------------------���濪ʼѵ��������------------------------------
            G_optimizer.zero_grad()
            G_seed_noise = gen_noise(Batch_size, Noise_feature).float()
            Fake_images = GENERATOR(G_seed_noise)
            Generator_Fake_logits = DISCRIMINATOR(Fake_images.view(Batch_size, 1, 28, 28))
            Generator_total_loss = G_loss(Generator_Fake_logits)
            # print('G_loss:', Generator_total_loss.detach().numpy())
            print("D_loss: {},   G_loss: {}".format(Discriminator_total_loss.detach().numpy(),
                                                    Generator_total_loss.detach().numpy()))
            Generator_total_loss.backward()
            G_optimizer.step()

            if Iter_count % show_every == 0:
                print('I love you.'*10)
                print(Discriminator_total_loss.detach().numpy())
                # print('Iter: {}, D: {:.4}, G:{:.4}'.format(Iter_count, Discriminator_total_loss.detach().numpy(),
                #                                            Generator_total_loss.detach().numpy()))
                # imgs_numpy = Fake_images.cpu().numpy()
                # show_images(imgs_numpy[0:16])
                # plt.show()
                # print()
            Iter_count += 1
        g_seed = gen_noise(128, Noise_feature).float()  # ���ﲻ������һ��ͼ������ӣ���������һ���Ĺ�һ�����̻����
        image = GENERATOR(g_seed).detach().numpy()[0].reshape((28, 28))  # --��BatchNorm1d���ܽ���batch=1��
        image_set.append(image)
    return np.array(image_set)


# a = gen_noise(32, 96)
# print(a)
