import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid

import gym
import cv2
import numpy as np
import random
from tensorboardX import SummaryWriter
import argparse


IMAGE_SIZE = 64
DISCR_FILTERS = 64
GENER_FILTERS = 64
LATENT_VECTOR_SIZE = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 100
SHOW_IMAGE_EVERY_ITER = 1000


'''
First we need to create an inputwrapper which includes several transformations
'''
class InputWrapper(gym.ObservationWrapper):
    def __init__(self, *args):
        super(InputWrapper, self).__init__(*args)
        # we will put several envs together and we want to check whether their obs are all Box
        assert isinstance(self.observation_space, gym.spaces.Box)
        # just in case we will transform all envs' obs to Box
        old_space = self.observation_space # actually self.observation is inherited from Wrapper which is the parent of ObservationWrapper

        """if shape=None low and high should have same shape. low and high should be ndarray
        for example obs = Box(np.array([-10, 0]), np.array([10, 1])) then call obs.sample() it gives: array([3.5775906 , 0.72063265])
        """
        self.observation_space = gym.spaces.Box(
            low=self._observation(old_space.low), 
            high=self._observation(old_space.high),
        )

    """then we need to redefine observation method in ObservationWrapper
    1. Resize the input image from 210×160 (the standard Atari resolution) to a square size 64×64Move the color plane of the image 
    2. from the last position to the first, to meet the PyTorch convention of convolution layers that input a tensor with the 
       shape of the channels, height, and width
    3. Cast the image from bytes to float
    this method should return the observation that will be given to the agent.
    actually in ObservationWrapper the step and reset method will call self.observation.
    """
    def _observation(self, observation):
        obs = cv2.resize(observation, (IMAGE_SIZE, IMAGE_SIZE))
        obs = np.moveaxis(obs, 2, 0)
        return obs.astype(np.float32) # obs and observation are all ndarray


class Discriminator(nn.Module):
    '''input_shape will be the atari window something like (H, W, C), different envs have different shape
    however we will let the net take in transformed inputs with shape (3, IMAGE_SIZE, IMAGE_SIZE)
    since we use InputWrapper and redefined _observation() method so env.reset() and env.step will return
    an obs with shape (3, 64, 64), so in_channels = input_shape[0] = 3s.
    '''
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        # this pipe converges image into the single number
        self.conv_pipe = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=DISCR_FILTERS,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS, out_channels=DISCR_FILTERS*2,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 2, out_channels=DISCR_FILTERS * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 4, out_channels=DISCR_FILTERS * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 8, out_channels=1,
                      kernel_size=4, stride=1, padding=0),
            nn.Sigmoid() # the probability that the Discriminator thinks our input image is from the real dataset.
        )

    def forward(self, x):
        conv_out = self.conv_pipe(x)
        '''Returns a tensor with all the dimensions of input of size 1 removed.
        For example, if input is of shape: (A×1×B×C×1×D) then 
        the out tensor will be of shape: (A×B×C×D) .
        torch.size([x]) mathematically a matrix [1, x]
        '''
        return conv_out.view(-1, 1).squeeze(dim=1)


class Generator(nn.Module):
    def __init__(self, output_shape):
        super(Generator, self).__init__()
        # pipe deconvolves input(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1) vector into (BATCH_SIZE, 3, 64, 64) image
        self.pipe = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LATENT_VECTOR_SIZE, out_channels=GENER_FILTERS * 8,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(GENER_FILTERS * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 8, out_channels=GENER_FILTERS * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 4, out_channels=GENER_FILTERS * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 2, out_channels=GENER_FILTERS,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS, out_channels=output_shape[0],
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.pipe(x)


'''
As input, we will use screenshots from several Atari games played simultaneously by a random agent and it is 
generated by the following function. This infinitely samples the environment from the provided array, 
issues random actions, and remembers observations in the batch list. When the batch becomes of the required size, 
we normalize the image, convert it to a tensor, and yield from the generator. The check for the non-zero mean of 
the observation is required due to a bug in one of the games to prevent the flickering of an image.
'''
def generate_inputs(envs, batch_size = BATCH_SIZE):
    inputs = [env.reset() for env in envs]
    env_gen = iter(lambda: random.choice(envs), None)

    while True:
        env = next(env_gen)
        obs, reward, done, _ = env.step(env.action_space.sample())
        if np.mean(obs) > 0.01:
            inputs.append(obs)
        if len(inputs) == batch_size:
            inputs_v = np.array(inputs, dtype=np.float32)
            inputs_v = inputs_v * 2.0 / 255.0 - 1.0 # normalize to [-1, 1]
            yield torch.tensor(inputs_v)
            inputs.clear()
        if done:
            env.reset()
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable Cuda Computation")
    args = parser.parse_args()

    torch.manual_seed(1)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    env_names = ("Breakout-v0", "AirRaid-v0", "Pong-v0")
    envs = [InputWrapper(gym.make(name)) for name in env_names]
    output_shape = input_shape = envs[0].observation_space.shape

    gen_net = Generator(output_shape)
    dis_net = Discriminator(input_shape)
    criterion = nn.BCELoss()
    gen_optimizer = optim.Adam(params=gen_net.parameters(), lr = LEARNING_RATE, betas=(0.5, 0.999))
    dis_optimizer = optim.Adam(params=dis_net.parameters(), lr = LEARNING_RATE, betas=(0.5, 0.999))

    train(envs, gen_net, dis_net, criterion, gen_optimizer, dis_optimizer, device)


def train(envs, gen_net, dis_net, criterion, gen_optimizer, dis_optimizer, device):
    gen_net.to(device)
    dis_net.to(device)
    true_labels = torch.ones(BATCH_SIZE, device=device)
    flase_labels = torch.zeros(BATCH_SIZE, device=device)
    gen_losses, dis_losses = [], []
    iter_n = 0
    writer = SummaryWriter(comment="gan")
    print("start training, using device: ", device)

    for dis_inputs in generate_inputs(envs):
        gen_inputs = torch.randn(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1).to(device)
        gen_outputs = gen_net(gen_inputs)

        # train discriminator
        dis_net.train()
        dis_inputs = dis_inputs.to(device) # can't change in place
        dis_optimizer.zero_grad()   
        dis_outputs_true = dis_net(dis_inputs)
        dis_outputs_false = dis_net(gen_outputs.detach())
        dis_loss = criterion(dis_outputs_true, true_labels) + criterion(dis_outputs_false, flase_labels)
        dis_loss.backward()
        dis_optimizer.step()
        dis_losses.append(dis_loss.item())
        # dis_optimizer.zero_grad()
        
        # train generator
        gen_net.train()
        gen_optimizer.zero_grad()
        adv_outputs = dis_net(gen_outputs)
        gen_loss = criterion(adv_outputs, true_labels)
        gen_loss.backward()
        gen_optimizer.step()
        gen_losses.append(gen_loss.item())
        # gen_optimizer.zero_grad()
        
        # dis_loss.backward()
        # gen_loss.backward()

        # dis_optimizer.step()
        # gen_optimizer.step()

        # monitor
        iter_n = iter_n + 1
        if iter_n % REPORT_EVERY_ITER == 0:
            print("gen_loss:%5.4f    dis_loss:%5.6f" % (np.mean(gen_losses), np.mean(dis_losses)))
            writer.add_scalar("gen_loss", np.mean(gen_losses), iter_n)
            writer.add_scalar("dis_loss", np.mean(dis_losses), iter_n)
            gen_losses.clear()
            dis_losses.clear()
        
        if iter_n % SHOW_IMAGE_EVERY_ITER == 0:
            writer.add_image("fake", make_grid(gen_outputs[:64], normalize=True), iter_n)
            writer.add_image("real", make_grid(dis_inputs[:64], normalize=True), iter_n)

        if iter_n == 40000:
            break

    print("done!")


if __name__ == "__main__":
    main()