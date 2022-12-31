from models import Generator_Discrminator
import torch
from torch import nn
import utils

z_dim =32
n_dim =16
gen =Generator(z_dim + n_dim).to(device)
disc =Discriminator(1).to(device)
# info =torch.load('/kaggle/input/frac-vector-01/frac_vector_info_01.pkl',map_location=device)
# gen.load_state_dict(info['generator_state'])
# disc.load_state_dict(info['discriminator_state'])

LR =0.0002
lr_decay_step =10
batch_size = 100
G_optimizer =optim.Adam(gen.parameters(),lr=LR)
D_optimizer =optim.Adam(disc.parameters(),lr=LR)
Info_optimizer =optim.Adam(itertools.chain(gen.parameters(),disc.parameters()),
                          lr =LR)
dataloader =DataLoader(train_data,batch_size=batch_size,shuffle=True)

c_n = nn.BCELoss()
c_c = nn.NLLLoss()

G_loss = []  # info['generator_loss']
D_loss = []  # info['discriminator loss']
Info_loss = []  # info['Info_loss']
epochs = 30
# Info_loss =[]
cur_step = 0
display_step = 500

for epoch in range(epochs):

    if epoch % lr_decay_step == 0 and epoch > 0:
        LR *= 0.5
        for param_group in G_optimizer.param_groups:
            param_group['lr'] = LR
        for param_group in D_optimizer.param_groups:
            param_group['lr'] = LR

    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        alpha = np.random.uniform(1., 1.5)
        frac_vector = generate_label(labels, alpha, n_class=z_dim).to(device)
        labels = labels.to(device)
        # frac_labels = frac_label(alpha,labels,n_class =10).to(device)
        noise = torch.randn(imgs.shape[0], n_dim).to(device)
        input = torch.cat((frac_vector, noise), 1).view(-1, z_dim + n_dim, 1, 1)
        fakes = gen(input)

        ### train the Discriminator
        D_optimizer.zero_grad()
        D_real_logits, _ = disc(imgs)
        D_fake_logits, _ = disc(fakes.detach())
        p_real = torch.squeeze(D_real_logits)
        p_fake = torch.squeeze(D_fake_logits)

        d_loss = -torch.mean(torch.log(p_real) + torch.log(1 - p_fake))
        d_loss.backward()
        D_optimizer.step()
        D_loss.append(d_loss.item())

        ### track the Generator
        G_optimizer.zero_grad()
        fake_logits, _ = disc(fakes)
        p_fake = torch.squeeze(fake_logits)
        g_loss = -torch.mean(torch.log(p_fake))
        g_loss.backward()
        G_optimizer.step()
        G_loss.append(g_loss.item())

        ####track the info_optimizer
        Info_optimizer.zero_grad()
        generated = gen(input)
        _, D_c = disc(imgs)
        _, F_c = disc(generated)
        D_c_loss = c_c(D_c, labels)
        F_c_loss = c_c(F_c, labels)
        info_loss = D_c_loss + F_c_loss
        info_loss.backward()
        Info_optimizer.step()
        Info_loss.append(info_loss.item())

        if cur_step % display_step == 0 and cur_step > 0:
            gen_mean = sum(D_loss[-display_step:]) / display_step
            disc_mean = sum(G_loss[-display_step:]) / display_step
            info_mean = sum(Info_loss[-display_step:]) / display_step
            print(f"Epoch {epoch} -> G_loss: {gen_mean},D_loss: {disc_mean},\
            Info_loss:{info_mean}")
            show_tensor_images(fakes)
            show_tensor_images(imgs)

            step_bins = 20
            # 每个元素有step_bins个，共有len(generator_losses) // step_bins个不同的元素
            x_axis = sorted([i * step_bins for i in range(len(G_loss) // step_bins)] * step_bins)
            num_examples = (len(G_loss) // step_bins) * step_bins  # 460
            plt.plot(
                range(num_examples // step_bins),
                torch.Tensor(G_loss[:num_examples]).view(-1, step_bins).mean(1),
                label="Generator Loss"
            )
            plt.plot(
                range(num_examples // step_bins),
                torch.Tensor(Info_loss[:num_examples]).view(-1, step_bins).mean(1),
                label="Q Loss"
            )
            plt.plot(
                range(num_examples // step_bins),
                torch.Tensor(D_loss[:num_examples]).view(-1, step_bins).mean(1),
                label="Discriminator Loss"
            )
            plt.legend()
            plt.show()
        cur_step += 1