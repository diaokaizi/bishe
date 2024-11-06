from . import MAE as mae
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
from fanogan_muti.train_wgangp import train_wgangp
from fanogan_muti.train_encoder_izif import train_encoder_izif
from model.gan import Generator, Discriminator, Encoder
from tools import SimpleDataset, load_UGR16, NormalizeTransform
import model.MAE as mae
import numpy as np
import pandas as pd
from fanogan_muti.test_anomaly_detection import test_anomaly_detection
import torch.autograd as autograd
import torch.nn as nn
from torch.utils.model_zoo import tqdm

class MAEGAN:

    def __init__(self, opt, input_dim, maxAE=10, feature_map=None, batch_size=64):
        # Parameters:
        self.opt = opt
        self.mae_model = mae.MAE(input_dim,maxAE,0,0, feature_map=feature_map)
        self.batch_size = batch_size
        self.feature_map = feature_map
        self.generator = None
        self.discriminator = None
        self.encoder = None
        self.gan_input_dim = None


    def train(self, data):
        print("Running KitNET:")
        mae_output = self.trainMAE(data)
        print("Running fanogan:")
        train_dataloader = self.load_gan_input(mae_output, label = np.zeros(len(mae_output)), is_train=True)
        self.gan_input_dim = mae_output.shape[1]
        latent_dim = int(self.gan_input_dim * 0.5)

        self.generator = Generator(self.gan_input_dim, latent_dim)
        self.discriminator = Discriminator(self.gan_input_dim)
        self.train_wgangp(train_dataloader, "cpu", latent_dim)

        self.encoder = Encoder(self.gan_input_dim, latent_dim)
        self.train_encoder_izif(train_dataloader, "cpu")

    def test(self, data, label):
        mae_output = self.testMAE(data)
        test_dataloader = self.load_gan_input(mae_output, label = label, is_train=False)
        return self.test_anomaly_detection(test_dataloader, "cpu")

    def load_gan_input(self, mae_output, label, is_train):
        mae_output = torch.from_numpy(mae_output).float()
        print(mae_output)
        print(mae_output.shape)

        if is_train:
            mean = mae_output.mean(axis=0)  # Mean of each feature
            std = mae_output.std(axis=0)
            normalize = NormalizeTransform(mean, std)
            torch.save({'mean': mean, 'std': std}, 'results/normalize_params.pt')
        else:
            normalize_params = torch.load('results/normalize_params.pt')
            mean = normalize_params['mean']
            std = normalize_params['std']
            # 创建 NormalizeTransform 对象
            normalize = NormalizeTransform(mean, std)
        dataset = SimpleDataset(mae_output, label, transform=normalize)
        train_dataloader = DataLoader(dataset, batch_size=self.batch_size,shuffle=False)
        return train_dataloader

    def trainMAE(self, data):
        if self.feature_map == None:
            for i in tqdm(range(data.shape[0])):
                self.mae_model.trainfm(data[i,]) #will train during the grace periods, then execute on all the rest.
            self.feature_map = self.mae_model.cluster()
            gan_input_dim = len(self.feature_map)
        print(self.feature_map)
        print(gan_input_dim)
        output = np.zeros([data.shape[0], gan_input_dim]) # a place to save the scores
        for i in tqdm(range(data.shape[0])):
            output[i] = self.mae_model.train(data[i,]) #will train during the grace periods, then execute on all the rest.
        self.mae_model.save("results")
        return output

    def testMAE(self, data):
        output = np.zeros([data.shape[0], self.gan_input_dim])
        for i in tqdm(range(data.shape[0])):
            output[i] = self.mae_model.execute(data[i,])
        return output
    
    def compute_gradient_penalty(self, real_samples, fake_samples, device):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        
        alpha = torch.rand(real_samples.shape[1], device=device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples)
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        d_interpolates = self.discriminator(interpolates)
        fake = torch.ones(*d_interpolates.shape, device=device)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                grad_outputs=fake, create_graph=True,
                                retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.shape[0], -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


    def train_wgangp(self, dataloader, device, latent_dim, lambda_gp=10):
        self.generator.to(device)
        self.discriminator.to(device)

        optimizer_G = torch.optim.Adam(self.generator.parameters(),
                                    lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))

        os.makedirs("results/images", exist_ok=True)

        padding_epoch = len(str(self.opt.n_epochs))
        padding_i = len(str(len(dataloader)))

        batches_done = 0
        for epoch in range(self.opt.n_epochs):
            for i, (gsa_input, _)in enumerate(dataloader):
                real_imgs = gsa_input.to(device)
                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Sample noise as generator input
                z = torch.randn(gsa_input.shape[0], latent_dim, device=device)

                # Generate a batch of images
                fake_imgs = self.generator(z)

                # Real images
                real_validity = self.discriminator(real_imgs)
                # Fake images
                fake_validity = self.discriminator(fake_imgs.detach())
                # Gradient penalty
                gradient_penalty = self.compute_gradient_penalty(
                                                            real_imgs.data,
                                                            fake_imgs.data,
                                                            device)
                # Adversarial loss
                d_loss = (-torch.mean(real_validity) + torch.mean(fake_validity)
                        + lambda_gp * gradient_penalty)

                d_loss.backward()
                optimizer_D.step()

                optimizer_G.zero_grad()

                # Train the generator and output log every n_critic steps
                if i % self.opt.n_critic == 0:

                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Generate a batch of images
                    fake_imgs = self.generator(z)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = self.discriminator(fake_imgs)
                    g_loss = -torch.mean(fake_validity)

                    g_loss.backward()
                    optimizer_G.step()

                    print(f"[Epoch {epoch:{padding_epoch}}/{self.opt.n_epochs}] "
                        f"[Batch {i:{padding_i}}/{len(dataloader)}] "
                        f"[D loss: {d_loss.item():3f}] "
                        f"[G loss: {g_loss.item():3f}]")

                    # if batches_done % opt.sample_interval == 0:
                    #     save_image(fake_imgs.data[:25],
                    #                f"results/images/{batches_done:06}.png",
                    #                nrow=5, normalize=True)

                    batches_done += self.opt.n_critic

            torch.save(self.generator.state_dict(), "results/generator")
            torch.save(self.discriminator.state_dict(), "results/discriminator")

    def train_encoder_izif(self, dataloader, device, kappa=1.0):
        self.generator.load_state_dict(torch.load("results/generator"))
        self.discriminator.load_state_dict(torch.load("results/discriminator"))

        self.generator.to(device).eval()
        self.discriminator.to(device).eval()
        self.encoder.to(device)

        criterion = nn.MSELoss()

        optimizer_E = torch.optim.Adam(self.encoder.parameters(),
                                    lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))

        os.makedirs("results/images_e", exist_ok=True)

        padding_epoch = len(str(self.opt.n_epochs))
        padding_i = len(str(len(dataloader)))

        batches_done = 0
        for epoch in range(self.opt.n_epochs):
            for i, (imgs, _) in enumerate(dataloader):

                # Configure input
                real_imgs = imgs.to(device)

                # ----------------
                #  Train Encoder
                # ----------------

                optimizer_E.zero_grad()

                # Generate a batch of latent variables
                z = self.encoder(real_imgs)

                # Generate a batch of images
                fake_imgs = self.generator(z)

                # Real features
                real_features = self.discriminator.forward_features(real_imgs)
                # Fake features
                fake_features = self.discriminator.forward_features(fake_imgs)

                # izif architecture
                loss_imgs = criterion(fake_imgs, real_imgs)
                loss_features = criterion(fake_features, real_features)
                e_loss = loss_imgs + kappa * loss_features

                e_loss.backward()
                optimizer_E.step()

                # Output training log every n_critic steps
                if i % self.opt.n_critic == 0:
                    print(f"[Epoch {epoch:{padding_epoch}}/{self.opt.n_epochs}] "
                        f"[Batch {i:{padding_i}}/{len(dataloader)}] "
                        f"[E loss: {e_loss.item():3f}]")

                    # if batches_done % opt.sample_interval == 0:
                    #     fake_z = encoder(fake_imgs)
                    #     reconfiguration_imgs = generator(fake_z)
                    #     save_image(reconfiguration_imgs.data[:25],
                    #                f"results/images_e/{batches_done:06}.png",
                    #                nrow=5, normalize=True)

                    batches_done += self.opt.n_critic
            torch.save(self.encoder.state_dict(), "results/encoder")


    def test_anomaly_detection(self, dataloader, device, kappa=1.0):
        self.generator.load_state_dict(torch.load("results/generator"))
        self.discriminator.load_state_dict(torch.load("results/discriminator"))
        self.encoder.load_state_dict(torch.load("results/encoder"))

        self.generator.to(device).eval()
        self.discriminator.to(device).eval()
        self.encoder.to(device).eval()

        criterion = nn.MSELoss()

        with open("results/score.csv", "w") as f:
            f.write("label,img_distance,anomaly_score,z_distance\n")
        results = []
        for (img, label) in tqdm(dataloader):

            real_img = img.to(device)

            real_z = self.encoder(real_img)
            fake_img = self.generator(real_z)
            fake_z = self.encoder(fake_img)

            real_feature = self.discriminator.forward_features(real_img)
            fake_feature = self.discriminator.forward_features(fake_img)

            # Scores for anomaly detection
            img_distance = criterion(fake_img, real_img)
            loss_feature = criterion(fake_feature, real_feature)
            anomaly_score = img_distance + kappa * loss_feature

            z_distance = criterion(fake_z, real_z)

            with open("results/score.csv", "a") as f:
                f.write(f"{label.item()},{img_distance},"
                        f"{anomaly_score},{z_distance}\n")
            results.append(anomaly_score)
        return results
