import gym
import gym.spaces
import cv2
import collections
import numpy as np


"""This wrapper presses the FIRE button in environments that require that for the game to start. 
In addition to pressing FIRE, this wrapper checks for several corner cases that are present in some games.
"""
class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        
    def _reset(self): # don't understand
        self.env.reset()
        obs, reward, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, reward, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


"""This wrapper combines the repetition of actions during K frames and pixels from two consecutive frames.
Making an action decision every K steps, where K is usually 4 or 3. On intermediate frames, the chosen 
action is simply repeated. This allows training to speed up significantly, as processing every frame with
an NN is quite a demanding operation, but the difference between consequent frames is usually minor.
"""
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def _step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        # choose max values by position from combined observations
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def _reset(self):
        # clear past frame buffer and init to first obs from inner env
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


"""This wrapper is to convert input observations from the emulator, which normally has a 
resolution of 210×160 or 250x160 pixels with RGB color channels, to a grayscale 84×84 image. It does this using 
a colorimetric grayscale conversion (which is closer to human color perception than a simple averaging 
of color channels), resizing the image, and cropping the top and bottom parts of the result.
"""
class ProcessFramed84(gym.ObservationWrapper):
    def __init__(self, env):
        super(ProcessFramed84, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(84, 84, 1)
        )
    
    def _observation(self, obs):
        return ProcessFramed84.process(obs)

    @staticmethod
    def process(frame): # can't use "self"
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution"
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_img = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        res = resized_img[18:102, :] # crop the top and bottom parts
        res = np.reshape(res, [84, 84, 1])
        return res.astype(np.uint8) # it is unint8!


"""This wrapper changes the shape of the observation from HWC (height, width, channel)
to the CHW (channel, height, width) format required by PyTorch. The input shape of the tensor
has a color channel as the last dimension, but PyTorch's convolution layers assume the color
channel to be the first dimension.
"""
class ImageToPytorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPytorch, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=new_shape
        )

    def _observation(self, obs):
        return np.moveaxis(obs, 2, 0)


"""This wrapper converts observation data from bytes to floats, 
and scales every pixel's value to the range [0.0...1.0]
"""
class ScaledFloatFrame(gym.ObservationWrapper):
    def _observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


"""This class creates a stack of subsequent frames along the first dimension and returns them as an observation. 
The purpose is to give the network an idea about the dynamics of the objects, such as the speed and direction of 
the ball in Pong or how enemies are moving. This is very important information, which it is not possible to obtain 
from a single image.
"""
class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        old_space = self.env.observation_space
        self.observation_space = gym.spaces.Box(
            low=old_space.low.repeat(n_steps, axis=0), # extend observation space shpae by using "repeat"
            high=old_space.high.repeat(n_steps, axis=0)
        )

    def _reset(self):
        # clear buffer
        self.buffer = np.zeros_like(self.observation_space.low, dtype=np.float32)
        return self._observation(self.env.reset())

    def _observation(self, obs):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = obs
        return self.buffer


"""Creates an environment by its name and applies all the required wrappers to it
Pay attention, the order can't be changed!!!
ScaledFloatFrame<BufferWrapper<ImageToPytorch<ProcessFrame84<FireResetEnv<MaxAndSkipEnv<env>>>>>>
it's like matryoshka, for example:
env._observation(obs) = ScaledFloatFrame._observation(
    BufferWrapper._observation(
        ImageToPytorch._observation(
            ProcessFrame84._observation(obs)
        )
    )
)
so for observation it will be resized and turned to grayscale images first and then be converted
to tensor, then put into buffer finally get scaled to [0.0, 1.0]
"""
def make_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFramed84(env)
    env = ImageToPytorch(env)
    env = BufferWrapper(env, 4)
    env = ScaledFloatFrame(env)
    
    return env