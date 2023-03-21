
from gymnasium import spaces, utils

from miniworld.entity import Box, Ball, Key, MeshEnt
from miniworld.miniworld import MiniWorldEnv
from miniworld.params import DEFAULT_PARAMS

from minigrid.envs.babyai.goto_custom import GoalObj, ObjectGoalController
from gymnasium.core import ObservationWrapper
import numpy as np
import math
import matplotlib.pyplot as plt


class CustomMiniWorldEnv(MiniWorldEnv, utils.EzPickle):
    def __init__(self, size=10, max_episode_steps=180, obs_width=160, obs_height=120, **kwargs):
        assert size >= 2
        self.size = size # TODO: this size variable is confusing...
        self.max_steps = max_episode_steps
        self.custom_env_type = "miniworld" # undo
        
        # Init objects
        self.init_objects()
        
        MiniWorldEnv.__init__(self, max_episode_steps=max_episode_steps, obs_width=obs_width, obs_height=obs_height, **kwargs)
        utils.EzPickle.__init__(
            self, size=size, max_episode_steps=max_episode_steps, **kwargs
        )

        # Allow only movement actions (left/right/forward/backwards)
        self.action_space = spaces.Discrete(4)
        # goal space
        self.create_goal_space()
    
    def get_goal_images(self):
        self.reset()
        for key, obj in self.goal_controller.object_dict.items():
            assert obj.agent_view is not None
            self.agent.pos = obj.agent_view['pos']
            self.agent.dir = obj.agent_view['dir']
            # WARNING:
            # TODO: This is super dodgy - if change how env is rendered, then this is not updated...
            rgb_img_partial = self.render_obs()
            rgb_img_partial = np.moveaxis(rgb_img_partial, -1, 0)
            # set image
            obj.set_goal_image(rgb_img_partial)
        #     # visualize
        #     from minigrid.utils.window import Window
        #     window = Window("minigrid")
        #     window.show(block=False)
        #     window.show_img(np.moveaxis(rgb_img_partial, 0, 2))
        #     plt.title(obj.get_string_description())
        # import pdb; pdb.set_trace()
        
    def step(self, action):
        assert self.action_space.contains(action)
        obs, reward, termination, truncation, info = super().step(action)

        # if self.near(self.goal_object):
        #     reward += self._reward()
        #     termination = True

        return obs, reward, termination, truncation, info
    
    def init_objects(self):
        object_dict, init_goal, eval_goals = self.get_init_object_info()
        self.goal_controller = ObjectGoalController(object_dict, init_goal, eval_goals)
        
    def init_rollout_objects(self):
        
        def init_miniworld_rollout_object(goal_obj):
            if 'mesh' in goal_obj.obj_type:
                assert goal_obj.obj_type.split('-')[1] in self.goal_controller.object_dict.keys()
            else:
                assert goal_obj.color + " " + goal_obj.obj_type in self.goal_controller.object_dict.keys()
            # create minigrid object
            if goal_obj.obj_type == "ball":
                miniworld_obj = Ball(goal_obj.color)
            elif goal_obj.obj_type == "key":
                miniworld_obj = Key(goal_obj.color)
            elif goal_obj.obj_type == "box":
                miniworld_obj = Box(goal_obj.color)
            elif 'mesh' in goal_obj.obj_type:
                mesh_name = goal_obj.obj_type.split('-')[1]
                miniworld_obj = MeshEnt(mesh_name=mesh_name, height=1)
            else:
                assert False
            # add object to env
            pos = np.array([goal_obj.position[0], 0.8, goal_obj.position[1]])
            self.place_entity(miniworld_obj, pos=pos, dir=3*math.pi/4)
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
    
    def get_dir_marker(self, radian):
        return (3, 0, radian * 360/ (2*math.pi))
        
    def plot_rollout(self, observations):
        # TODO: add pos to info, and use infos...
        assert len(observations.shape) == 2
        assert observations.shape[1] == 3
        # print(observations.shape)
        
        fig = plt.figure()
        plt.xlim(0, self.env_cols)
        plt.ylim(-self.env_rows, 0)
        
        for key, obj in self.goal_controller.object_dict.items():
            if key != "empty":
                i, j = obj.position[0], obj.position[1]
                color = obj.plot_color
                plt.plot([i], [-j], marker="o", markersize=20, markerfacecolor=color)

        traj_x = observations[:, 1]
        traj_y = - observations[:, 2] # Note the minus!!
        time_step = - np.arange(observations.shape[0])
        
        plt.plot(traj_x, traj_y)
        plt.scatter(traj_x, traj_y, c=time_step, cmap='gray')
        
        # def get_dir_marker() # TODO
        end_dir = int(np.rint(observations[-1, 0]))
        plt.scatter([traj_x[-1]], [traj_y[-1]], marker=self.get_dir_marker(end_dir), s=150, color='red')
        
        return fig
    

class OneRoomCustom(CustomMiniWorldEnv):
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

    def __init__(self, size=10, max_episode_steps=180, obs_width=160, obs_height=120, **kwargs):
        super().__init__(size=size, max_episode_steps=max_episode_steps, obs_width=obs_width, obs_height=obs_height, **kwargs)

        
    def create_goal_space(self):
        self.rew_dist_thresh = max(1.5, 1.5 * self.max_forward_step)
        self.max_dist_to_goal = np.sqrt(2 * (self.size - 2)**2)
        self.coord_goal_space = spaces.Box(low=np.array([1, 1]), high=np.array([self.size-1, self.size-1]), dtype=np.int64)
        # TODO: fix this as bounds may be too tight/small
        self.env_rows = self.size + 1 # self.num_rows * (self.room_size-1)
        self.env_cols = self.size + 1 # self.num_cols * (self.room_size-1)

    def _gen_world(self):
        self.add_rect_room(min_x=0, max_x=self.size, min_z=0, max_z=self.size)

        mid_pos = round(self.size/2)
        self.place_agent(room=None, dir=None, min_x=mid_pos, max_x=mid_pos, min_z=mid_pos, max_z=mid_pos)
        
        self.init_rollout_objects()
        
        # self.goal_object = self.goal_controller.get_init_goal().minigrid_object # TODO: this is bad
        self.goal_controller.verify_object_dict()
    
    def get_init_object_info(self):
        self.indent = 1
        self.indent_far = self.size - self.indent
        # store list
        object_dict = {
                    "green ball": GoalObj("green", "ball", (self.indent, self.indent_far), {'pos': (2.5, 0.0, 7.64), 'dir': 3.76}),
                    "yellow key": GoalObj("yellow", "key", (self.indent_far, self.indent_far), {'pos': (7.34, 0.0, 7.47), 'dir': 5.56}),
                    "blue box": GoalObj("blue", "box", (self.indent_far, self.indent), {'pos': (7.43, 0.0, 2.47), 'dir': 0.87}),
                    "red key": GoalObj("red", "key", (self.indent, self.indent), {'pos': (2.34, 0.0, 2.36), 'dir': 2.42})
                    }
        object_dict["empty"] = GoalObj("empty", "empty", (int(self.size/2), int(self.size/2)), {'pos': (int(self.size/2), 0.0, int(self.size/2)), 'dir': 0})
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

    
class FourRoomsCustom(CustomMiniWorldEnv):
    """
    ## Description

    Classic four rooms environment. The goal is to reach the red box to get a
    reward in as few steps as possible.

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
    env = gym.make("MiniWorld-FourRooms-v0")
    ```

    """

    def __init__(self, size=10, max_episode_steps=100, random_init=False, obs_width=160, obs_height=120, **kwargs):
        self.init_agent_pos = np.array([3, 0.0, 3])
        self.random_init = random_init
        
        super().__init__(size=size, max_episode_steps=max_episode_steps, obs_width=obs_width, obs_height=obs_height, **kwargs)

    def _gen_world(self):
        ## Easier for all coords to be positive (for info tracking)
        # Top-left room
        room0 = self.add_rect_room(min_x=0, max_x=6, min_z=8, max_z=14, wall_tex="brick_wall")
        # Top-right room
        room1 = self.add_rect_room(min_x=8, max_x=14, min_z=8, max_z=14, wall_tex='lava') # floor_tex="grass")
        # Bottom-right room
        room2 = self.add_rect_room(min_x=8, max_x=14, min_z=0, max_z=6, wall_tex="wood")
        # Bottom-left room
        room3 = self.add_rect_room(min_x=0, max_x=6, min_z=0, max_z=6)

        # Add openings to connect the rooms together
        self.connect_rooms(room0, room1, min_z=10, max_z=12, max_y=2.2)
        self.connect_rooms(room1, room2, min_x=10, max_x=12, max_y=2.2)
        self.connect_rooms(room2, room3, min_z=2, max_z=4, max_y=2.2)
        self.connect_rooms(room3, room0, min_x=2, max_x=4, max_y=2.2)

        self.init_rollout_objects()
        # self.goal_object = self.goal_controller.get_init_goal().minigrid_object # TODO: this is bad
        self.goal_controller.verify_object_dict()
        # self.box = self.place_entity(Box(color="red"))
        
        if self.random_init:
            self.place_agent()
        else:
            self.place_entity(self.agent, pos=self.init_agent_pos, dir=155*(360/2*math.pi))
        

    def get_init_object_info(self):
        # store list
        object_dict = {
                    "cone": GoalObj(None, "mesh-cone", (0.5, 8.5), {'pos': (2, 0.0, 10), 'dir': 2.4}, plot_color='orange'), # room 0
                    "red key": GoalObj("red", "key", (5.5, 13.5), {'pos': (4, 0.0, 12), 'dir': 5.5}), # room 0
                    "blue box": GoalObj("blue", "box", (8.5, 8.5), {'pos': (10, 0.0, 10), 'dir': 2.35}), # room 1
                    "green ball": GoalObj("green", "ball", (13.5, 13.5), {'pos': (12, 0.0, 12), 'dir': 5.5}), # room 1
                    "duckie": GoalObj(None, "mesh-duckie", (8.5, 0.5), {'pos': (9, 0.0, 3), 'dir': 1.57}, plot_color='yellow'), # room 2
                    "office_chair": GoalObj(None, "mesh-office_chair", (13, 5), {'pos': (12, 0.0, 4), 'dir': 5.5}, plot_color='darkred'), # room 2
                    "barrel": GoalObj(None, "mesh-barrel", (0.5, 0.5), {'pos': (2, 0.0, 2), 'dir': 2.35}, "c"), # room 3
                    "office_desk": GoalObj(None, "mesh-office_desk", (5.5, 5.5), {'pos': (4, 0.0, 4), 'dir': 5.5}, plot_color='brown'), # room 3
                    }
        object_dict["empty"] = GoalObj("empty", "empty", (11, 11), {'pos': (11, 0, 11), 'dir': 0})
        # set init goal
        init_goal = "green ball" # green circle = hard, red key = medium
        eval_goals = [
                "green ball",
                "blue box",
                "cone",
                "barrel",
                "duckie",
            ]
        return object_dict, init_goal, eval_goals
    
    def create_goal_space(self):
        self.overall_size = 14
        self.rew_dist_thresh = max(2, 1.5 * self.max_forward_step)
        self.max_dist_to_goal = np.sqrt(2 * (self.overall_size - 2)**2)
        self.coord_goal_space = spaces.Box(low=np.array([0, 0]), high=np.array([self.overall_size, self.overall_size]), dtype=np.int64)
        # TODO: fix this as bounds may be too tight/small
        self.env_rows = self.overall_size + 1 # self.num_rows * (self.room_size-1)
        self.env_cols = self.overall_size + 1 # self.num_cols * (self.room_size-1)



class OneRoomS6Custom(OneRoomCustom):
    def __init__(self, size=10, max_episode_steps=100, **kwargs):
        super().__init__(size=size, max_episode_steps=max_episode_steps, **kwargs)


# Parameters for larger movement steps, fast stepping
default_params = DEFAULT_PARAMS.no_random()
default_params.set("forward_step", 0.7)
default_params.set("turn_step", 45)


class OneRoomS6FastCustom(OneRoomS6Custom):
    def __init__(
        self, max_episode_steps=50, size=10, params=default_params, domain_rand=False, **kwargs
    ):

        super().__init__(
            max_episode_steps=max_episode_steps,
            params=params,
            domain_rand=domain_rand,
            size=size,
            **kwargs
        )
        
class FourRoomsFastCustom(FourRoomsCustom):
    def __init__(
        self, random_init=False, size=10, params=default_params, domain_rand=False, **kwargs
    ):
        if random_init:
            max_episode_steps = 50
        else:
            max_episode_steps = 100

        super().__init__(
            max_episode_steps=max_episode_steps,
            params=params,
            domain_rand=domain_rand,
            size=size,
            random_init=random_init,
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
        state_space = spaces.Box(low=np.array([0, 1, 1]), high=np.array([2*math.pi, env.size-1, env.size-1]), dtype=np.int64) # TODO: verify radians high
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
        agent_dir = self.agent.dir % (2 * math.pi)
        obs_dict['state'] = np.array([agent_dir, self.agent.pos[0], self.agent.pos[2]])
        
        # Image
        if self.include_image:
            if self.agent_pov:
                img = obs # self.render_obs()
            else:
                img = self.render_top_view()
            obs_dict['image'] = np.moveaxis(img, -1, 0)
                
            assert obs_dict['image'].shape == self.observation_space['image'].shape

        return obs_dict