
from gymnasium import spaces, utils

from miniworld.entity import Box, Ball, Key
from miniworld.miniworld import MiniWorldEnv
from miniworld.params import DEFAULT_PARAMS

from minigrid.envs.babyai.goto_custom import GoalObj, ObjectGoalController
from gymnasium.core import ObservationWrapper
import numpy as np
import math


class OneRoomCustom(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Environment in which the goal is to go to a red box placed randomly in one big room.
    The `OneRoom` environment has two variants. The `OneRoomS6` environment gives you
    a room with size 6 (the `OneRoom` environment has size 10). The `OneRoomS6Fast`
    environment also is using a room with size 6, but the turning and moving motion
    is larger.

    ## Action Space

    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing a RGB image of what the agents sees.

    ## Rewards:

    +(1 - 0.2 * (step_count / max_episode_steps)) when red box reached and zero otherwise.

    ## Arguments

    ```python
    env = gym.make("MiniWorld-OneRoom-v0")
    # or
    env = gym.make("MiniWorld-OneRoomS6-v0")
    # or
    env = gym.make("MiniWorld-OneRoomS6Fast-v0")
    ```

    """

    def __init__(self, size=10, max_episode_steps=180, **kwargs):
        assert size >= 2
        self.size = size
        self.max_steps = max_episode_steps
        
        # Init objects
        self.init_objects()

        MiniWorldEnv.__init__(self, max_episode_steps=max_episode_steps, **kwargs)
        utils.EzPickle.__init__(
            self, size=size, max_episode_steps=max_episode_steps, **kwargs
        )

        # Allow only movement actions (left/right/forward)
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)
        # goal space
        self.create_goal_space()
        
        
    def create_goal_space(self):
        self.rew_dist_thresh = 2
        self.max_dist_to_goal = np.sqrt(2 * (self.size - 2)**2)
        self.coord_goal_space = spaces.Box(low=np.array([1, 1]), high=np.array([self.size-1, self.size-1]), dtype=np.int64)
        # TODO: fix this as bounds may be too tight/small

    def _gen_world(self):
        self.add_rect_room(min_x=0, max_x=self.size, min_z=0, max_z=self.size)

        self.place_agent()
        print("\nCHECK set same agent pos each time\n")
        
        self.init_rollout_objects()
        
        self.goal_object = self.goal_controller.get_init_goal().minigrid_object # TODO: this is bad
        self.goal_controller.verify_object_dict()

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        if self.near(self.goal_object):
            reward += self._reward()
            termination = True

        return obs, reward, termination, truncation, info
    
    def init_objects(self):
        object_dict, init_goal, eval_goals = self.get_init_object_info()
        self.goal_controller = ObjectGoalController(object_dict, init_goal, eval_goals)
    
    def get_init_object_info(self):
        self.indent = 1
        self.indent_far = self.size - self.indent
        # store list
        object_dict = {
                    "green ball": GoalObj("green", "ball", (self.indent, self.indent_far)),
                    "yellow key": GoalObj("yellow", "key", (self.indent_far, self.indent_far)),
                    "blue box": GoalObj("blue", "box", (self.indent_far, self.indent)),
                    "red key": GoalObj("red", "key", (self.indent, self.indent))
                    }
        # set init goal
        init_goal = "green ball"
        eval_goals = [
                "green ball",
                "yellow key",
                "blue box",
                "red key",
                "empty"
            ]
        return object_dict, init_goal, eval_goals
    
    
    def init_rollout_objects(self):
        
        def init_miniworld_rollout_object(goal_obj):
            assert goal_obj.color + " " + goal_obj.obj_type in self.goal_controller.object_dict.keys()
            # create minigrid object
            if goal_obj.obj_type == "ball":
                miniworld_obj = Ball(goal_obj.color)
            elif goal_obj.obj_type == "key":
                miniworld_obj = Key(goal_obj.color)
            elif goal_obj.obj_type == "box":
                miniworld_obj = Box(goal_obj.color)
            else:
                assert False
            # add object to env
            pos = (goal_obj.position[0], 0.8, goal_obj.position[1])
            self.place_entity(miniworld_obj, pos=pos)
            goal_obj.assign_minigrid_object(miniworld_obj)
        
        # iterate through objects    
        for key, obj in self.goal_controller.object_dict.items():
            if key != "empty":
                init_miniworld_rollout_object(obj)
        
        
    def get_agent_achieved_2D_pos(self):
        return (self.agent.pos[0], self.agent.pos[2])
        
    def set_to_train_mode(self):
        self.goal_controller.set_mode('train')
        
    def set_to_eval_mode(self):
        self.goal_controller.set_mode('eval')
    
    def init_dir_markers(self):
        assert False
        self.dir_markers = [">", "v", "<", "^"]
        
    def plot_rollout(self, observations):
        # TODO: add pos to info, and use infos...
        assert False


class OneRoomS6Custom(OneRoomCustom):
    def __init__(self, size=6, max_episode_steps=100, **kwargs):
        super().__init__(size=size, max_episode_steps=max_episode_steps, **kwargs)


# Parameters for larger movement steps, fast stepping
default_params = DEFAULT_PARAMS.no_random()
default_params.set("forward_step", 0.7)
default_params.set("turn_step", 45)


class OneRoomS6FastCustom(OneRoomS6Custom):
    def __init__(
        self, max_episode_steps=50, size=6, params=default_params, domain_rand=False, **kwargs
    ):

        super().__init__(
            max_episode_steps=max_episode_steps,
            params=params,
            domain_rand=domain_rand,
            **kwargs
        )
        
class CustomMiniworldDictObsWrapper(ObservationWrapper):

    def __init__(self, env, include_image=True, agent_pov=False):
        super().__init__(env)
        
        self.include_image = include_image
        self.agent_pov = agent_pov
        
        ## Create Obs space
        self.obs_space_dict = {}
        
        # [direction (radians), pos[0], pos[1]]
        state_space = spaces.Box(low=np.array([0, 1, 1]), high=np.array([2*math.pi, env.size-1, env.size-1]), dtype=np.int64)
        self.obs_space_dict['state'] = state_space

        # Images
        if self.include_image:
            obs_shape = env.observation_space.shape # already returning image observations
            new_image_space = spaces.Box(
                low=0,
                high=255,
                shape=(3, obs_shape[0], obs_shape[1]),
                dtype="int64",
            )
            self.obs_space_dict['image'] = new_image_space
        
        self.observation_space = spaces.Dict(self.obs_space_dict) # TODO: update to include more info in state?

    def observation(self, obs):
        obs_dict = {}
        # States
        assert len(self.agent.pos) == 3
        assert self.agent.pos[1] == 0.0
        obs_dict['state'] = np.array([self.agent.dir, self.agent.pos[0], self.agent.pos[2]])
        
        # Image
        if self.include_image:
            if self.agent_pov:
                img = obs # self.render_obs()
            else:
                img = self.render_top_view()
            obs_dict['image'] = np.moveaxis(img, -1, 0)
                
            assert obs_dict['image'].shape == self.obs_space_dict['image'].shape

        return obs_dict