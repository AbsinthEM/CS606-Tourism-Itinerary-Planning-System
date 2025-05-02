import os

import gymnasium as gym
import numpy as np
import numpy.random as rnd
from operators import *
from rcjsp import SMJSP, Parser
from src.alns import ALNS
from src.settings import DATA_PATH
from operators import *
from disruptions import *

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


class SMJSPAlnsEnv(gym.Env):
    def __init__(self, config, **kwargs):
        # Parameters
        self.config = config["environment"]

        self.psp = None
        self.rnd_state = None
        self.tourist_id = self.config['tourist_id'] if "tourist_id" in self.config.keys() else None
        # print("Tourist ID ", self.tourist_id)
        self.initial_solution = None
        self.best_solution = None
        self.current_solution = None

        # // Add code here to include other states that require reset
        # --------------------- Provided to students
        self.improvement = None
        self.current_improved = None

        # // Add aditional attributes if required to store additional state observations
        # ---------------------------------------

        self.cost_difference_from_best = None
        self.current_updated = None

        # # Simulated annealing acceptance criteria
        self.max_temperature = 5
        self.temperature = 5

        # Gym Environment Parameters
        self.reward = 0  # Total episode reward
        self.done = False  # Termination
        self.episode = 0  # Episode number (one episode consists of ngen generations)
        self.iteration = 0  # Current gen in the episode
        self.max_iterations = self.config[
            "iterations"
        ]  # max number of generations in an episode

        ### RL states improvements
        self.temperature_history = []
        self.prev_objectives = []
        self.recent_accepts = []
        self.episode_reward = 0
        self.reward_history = []
        self.improvement_streak = 0

        self.destroy_operator_usage = {}
        self.repair_operator_usage = {}
        self.pair_operator_usage = {}
        self.destroy_operator_success = {}
        self.repair_operator_success = {}
        self.pair_operator_success = {}
        
        self.prev_solutions = []
        self.operator_reward_matrix = np.zeros((20, 20))  # Max operator size


        # Defining Action and Observation Space
        # ----------------------------------------------------------------------
        # // Modify action and observation space as you see fit
        # self.action_space = gym.spaces.MultiDiscrete([5, 5, 10, 100])
        # gym.spaces.MultiDiscrete([{no of repair operators}, {no of destroy operators}, {no. of destroy factor intervals}])
        self.action_space = gym.spaces.MultiDiscrete([5, 6, 10])
        # self.observation_space = gym.spaces.Box(shape=(8,), low=0, high=100, dtype=np.float64)
        self.observation_space = gym.spaces.Box(
            shape=(16,), low=-1e5, high=1e5, dtype=np.float64
        )
        # ----------------------------------------------------------------------

    def make_observation(self):
        """
        Return the environment's current state
        """

        is_current_best = 0
        if self.current_solution.objective() == self.best_solution.objective():
            is_current_best = 1
        
         ### Add RL states improvements
        # acceptance ratio (last 20 steps)
        acceptance_ratio = np.mean(self.recent_accepts) if self.recent_accepts else 0.0
        # objective volatility (standard deviation)
        objective_std = np.std(self.prev_objectives) if len(self.prev_objectives) > 1 else 0.0
        # normalized temperature
        temp_normalized = self.temperature / self.max_temperature
        avg_reward = np.mean(self.reward_history) if self.reward_history else 0.0

        # Calculate current operator pair success rate
        d_idx, r_idx = self.last_action if hasattr(self, 'last_action') else (0, 0)
        d_name = self.dr_alns.destroy_operators[d_idx][0] if self.dr_alns else ""
        r_name = self.dr_alns.repair_operators[r_idx][0] if self.dr_alns else ""
        pair_key = (d_name, r_name)
        success_rate = 0.0
        if pair_key in self.pair_operator_usage and self.pair_operator_usage[pair_key] > 0:
            success_rate = self.pair_operator_success[pair_key] / self.pair_operator_usage[pair_key]
        
        # Calculate operator usage entropy
        usage_matrix = self.operator_usage + 1e-6
        prob_dist = usage_matrix / np.sum(usage_matrix)
        usage_entropy = -np.sum(prob_dist * np.log(prob_dist))

        # New features
        remaining_horizon = (self.max_iterations - self.iteration) / self.max_iterations
        best_gap = (self.initial_objective - self.best_solution.objective()) / (self.initial_objective + 1e-6)
        slack_ratio = self.psp.get_slack_ratio()
        diversity_score = self.psp.calculate_solution_diversity(self.prev_solutions)
        avg_pair_reward = self.operator_reward_matrix[d_idx, r_idx]

        # state = np.array(
        #     [self.improvement, self.cost_difference_from_best, is_current_best, self.temperature,
        #      self.stagcount, self.iteration / self.max_iterations, self.current_updated, self.current_improved],
        #     dtype=np.float64).squeeze()

        # ---------------------
        # // Add additional state observations here
        state = np.array(
            [
                self.improvement,
                self.cost_difference_from_best,
                is_current_best,
                self.stagcount,
                self.iteration / self.max_iterations,
                self.current_updated,
                self.current_improved,
                self.temperature,
                temp_normalized,
                acceptance_ratio,
                objective_std,
                self.episode_reward / (self.iteration + 1),
                success_rate,
                self.improvement_streak,
                avg_reward,
                usage_entropy,
            #     remaining_horizon,
            #     best_gap,
            #     slack_ratio,
            #     diversity_score,
            #     avg_pair_reward
            ],  # State observations
            dtype=np.float64,
        ).squeeze()
        # ---------------------

        return state

    def reset(self, seed=None, options=None, run = 0):
        """
        The reset method: returns the current state of the environment (first state after initialization/reset)
        """

        # We choose a random tourist from the parser
        if seed == None:
            seed = 606

        self.rnd_state = rnd.RandomState(seed)

        parsed = Parser("AttractionProfile.csv", "TouristProfile.csv")

        chosen_tourist = self.rnd_state.choice(parsed.tourists, 1)[0]

        if self.tourist_id:
            print("Choosing Tourist")
            for idx, c_tourist in enumerate(parsed.tourists):
                if int(c_tourist.idx) == self.tourist_id:
                    print("Chosen")
                    chosen_tourist = parsed.tourists[idx]
                    break

        for a_ in parsed.attractions:
            new_opening = {}
            for dayk, hourset in a_.opening_hours.items():
                new_opening[dayk-1] = hourset
            a_.opening_hours = new_opening

        # print("Chosen Tourist ID ", chosen_tourist.idx)
        self.chosen_tourist = chosen_tourist

        # Apply random disruptions
        disruptions = [
            rainy_day_attraction,
            early_closure,
            nothing_happens_attraction,
            heat_wave_attraction
        ]
        self.disrupt_names = []
        for day in range(self.chosen_tourist.days):
            random_disruption = self.rnd_state.choice(disruptions, 1)[0]
            parsed.attractions = random_disruption(parsed.attractions, day)
            self.disrupt_names.append(random_disruption.__name__)

        psp = SMJSP(chosen_tourist, parsed.attractions)

        init_objective = psp.random_initialize(seed)

        self.psp = psp
        self.initial_solution = psp
        self.initial_objective = init_objective
        self.current_solution = copy.deepcopy(self.initial_solution)
        self.best_solution = copy.deepcopy(self.initial_solution)

        # Adding of Destroy and Repair Operators
        # // You should import and add your operators.py functions here
        self.dr_alns = ALNS(self.rnd_state)
        self.dr_alns.add_destroy_operator(destroy_random, "Random")
        self.dr_alns.add_destroy_operator(destroy_worst, "Worst")
        self.dr_alns.add_destroy_operator(destroy_day, "Day")
        self.dr_alns.add_destroy_operator(destroy_preference, "Preference")
        self.dr_alns.add_destroy_operator(destroy_cluster, "Cluster")
        self.dr_alns.add_destroy_operator(destroy_most_frequent_category, "Most Frequent Category")
        self.dr_alns.add_destroy_operator(destroy_recently_added, "Recently Added")
        self.dr_alns.add_destroy_operator(destroy_expensive, "Expensive")

        self.dr_alns.add_repair_operator(repair_greedy, "Greedy")
        self.dr_alns.add_repair_operator(repair_regret, "Regret")
        self.dr_alns.add_repair_operator(repair_random, "Random")
        self.dr_alns.add_repair_operator(repair_balanced, "Balanced")
        self.dr_alns.add_repair_operator(repair_nearest_neighbor, "Nearest")
        self.dr_alns.add_repair_operator(repair_rainy, "Repair Rainy")
        self.dr_alns.add_repair_operator(repair_maximize_attractions, "Maximize Attractions")
        self.dr_alns.add_repair_operator(repair_cheapest_first, "Cheapest First")
        self.dr_alns.add_repair_operator(repair_diversity_first, "Diversity First")
        self.dr_alns.add_repair_operator(repair_time_slot_fit, "Time Slot Fit")
        self.dr_alns.add_repair_operator(repair_cheapest_fun_proximity, "Cheapest Fun Proximity")

        # reset tracking values
        # // Add code here to reset the additional
        # observation states that you defined
        self.stagcount = 0
        self.current_improved = 0
        self.current_updated = 0
        self.episode += 1
        self.temperature = self.max_temperature
        self.improvement = 0
        self.cost_difference_from_best = 0

        self.iteration, self.reward = 0, 0
        self.done = False

        ### RL states improvements
        self.temperature_history = []
        self.prev_objectives = []
        self.operator_usage = np.zeros((len(self.dr_alns.destroy_operators), len(self.dr_alns.repair_operators)))
        self.recent_accepts = []
        self.episode_reward = 0
        self.reward_history = []
        self.improvement_streak = 0
        self.prev_solutions = []
        self.operator_reward_matrix = np.zeros((len(self.dr_alns.destroy_operators), len(self.dr_alns.repair_operators)))


        # Set up operator usage/success tracking dictionaries
        destroy_names = [name for name, _ in self.dr_alns.destroy_operators]
        repair_names = [name for name, _ in self.dr_alns.repair_operators]

        self.destroy_operator_usage = {name: 0 for name in destroy_names}
        self.repair_operator_usage = {name: 0 for name in repair_names}
        self.pair_operator_usage = {
            (d, r): 0 for d in destroy_names for r in repair_names
        }
        self.destroy_operator_success = {name: 0 for name in destroy_names}
        self.repair_operator_success = {name: 0 for name in repair_names}
        self.pair_operator_success = {
            (d, r): 0 for d in destroy_names for r in repair_names
        }

        if self.episode % 1000 == 0 and len(self.episode_rewards) >= 1000:
            avg_reward = np.mean(self.episode_rewards[-1000:])
            print(f"[Episode {self.episode}] Avg Reward (last 1000 eps): {avg_reward:.2f} | Best Objective: {self.best_solution.objective():.2f}")
        return self.make_observation(), {}
        

    def step(self, action):
        self.iteration += 1
        self.stagcount += 1
        self.current_updated = 0
        self.reward = 0
        self.improvement = 0
        self.cost_difference_from_best = 0
        self.current_improved = 0
        # // Add code here to "step" the additional
        # observation states that you defined

        current = self.current_solution
        best = self.best_solution

        d_idx, r_idx = action[0], action[1]
        d_name, d_operator = self.dr_alns.destroy_operators[d_idx]

        factors = {
            0: 0.1,
            1: 0.2,
            2: 0.3,
            3: 0.4,
            4: 0.5,
            5: 0.6,
            6: 0.7,
            7: 0.8,
            8: 0.9,
            9: 1.0,
        }
        destory_factor = factors[action[2]]
        # self.temperature = (1/(action[3]+1)) * self.max_temperature

        destroyed = d_operator(current, self.rnd_state, destory_factor)

        r_name, r_operator = self.dr_alns.repair_operators[r_idx]
        candidate = r_operator(destroyed, self.rnd_state)

        new_best, new_current = self.consider_candidate(best, current, candidate)

        self.reward_and_update(new_best, best, new_current, current)

        self.cost_difference_from_best = - (
            self.current_solution.objective() / self.best_solution.objective()
        ) * 100

        ### Add RL states improvements
        self.operator_usage[d_idx, r_idx] += 1

        accepted = int(self.current_updated)
        self.recent_accepts.append(accepted)
        if len(self.recent_accepts) > 20:
            self.recent_accepts.pop(0)

        self.prev_objectives.append(self.current_solution.objective())
        if len(self.prev_objectives) > 5:
            self.prev_objectives.pop(0)

        self.episode_reward += self.reward
       
        # Track running average reward
        self.reward_history.append(self.reward)
        if len(self.reward_history) > 10:
            self.reward_history.pop(0)

        # Track improvement streak
        if self.improvement == 1:
            self.improvement_streak += 1
        else:
            self.improvement_streak = 0

        # Update operator usage and success counts
        d_name = self.dr_alns.destroy_operators[d_idx][0]
        r_name = self.dr_alns.repair_operators[r_idx][0]
        self.last_action = (d_idx, r_idx)
        self.destroy_operator_usage[d_name] += 1
        self.repair_operator_usage[r_name] += 1
        self.pair_operator_usage[(d_name, r_name)] += 1

        if self.current_improved == 1:
            self.destroy_operator_success[d_name] += 1
            self.repair_operator_success[r_name] += 1
            self.pair_operator_success[(d_name, r_name)] += 1
        
        self.operator_reward_matrix[d_idx, r_idx] = (
            0.9 * self.operator_reward_matrix[d_idx, r_idx] + 0.1 * self.reward
        )

        self.prev_solutions.append(self.current_solution)
        if len(self.prev_solutions) > 5:
            self.prev_solutions.pop(0)

        state = self.make_observation()

        # Check if episode is finished (max ngen per episode)
        if self.iteration == self.max_iterations:
            self.done = True

        return state, self.reward, self.done, False, {}

    def reward_and_update(self, new_best, best, new_current, current):
        # ------------------------------------------------------------
        # // Modify Reward Function Here as you see fit
        if new_best != best and new_best is not None:
            # found new best solution
            self.best_solution = new_best
            self.current_solution = new_best
            self.current_updated = 1
            self.reward += 5
            self.stagcount = 0
            self.current_improved = 1

        elif new_current != current and new_current.objective() > current.objective():
            # solution accepted
            self.current_solution = new_current
            self.current_updated = 1
            self.current_improved = 1

        elif new_current != current and new_current.objective() <= current.objective():
            self.current_solution = new_current
            self.current_updated = 1

        if new_current.objective() > current.objective():
            self.improvement = 1
        # ------------------------------------------------------------

    # def consider_candidate(self, best, curr, cand):
    #     # -----------------------------------------------------
    #     # // Modify acceptance criteria as you see fit
    #     # Hill Climbing
    #     if cand.objective() < best.objective():
    #         return cand, cand
    #     else:
    #         return None, curr

    #     # // You could try other strategies like:
    #     # 1. Simulated Annealing ?
    #     # 2. Record to Record Travel ?
    #     # ------------------------------------------------------

    def consider_candidate(self, best, curr, cand):
        # Simulated Annealing
    
        diff = curr.objective() - cand.objective()
        probability = np.exp(diff / self.temperature)
        if cand.objective() < best.objective():
            return cand, cand
    
        # accepted:
        elif probability >= rnd.random():
            return None, cand
    
        else:
            return None, curr

    # --------------------------------------------------------------------------------------------------------------------

    def run(self, model, seed = None, episodes = 1):
        """
        Use a trained model to select actions.
        """
        try:
            for episode in range(episodes):
                self.done = False
                state, _ = self.reset(seed = seed, run = 1)

                while not self.done:
                    state = np.array(state)
                    action, _ = model.predict(state, deterministic=True)
                    state, reward, self.done, _, info = self.step(action)

                    # Optional: Debugging/logging
                    print(f"State: {state}, Reward: {reward}, Done: {self.done}")

        except KeyboardInterrupt:
            print("Execution interrupted. Stopping.")

    def sample(self):
        """
        Sample random actions and run the environment
        """
        for episode in range(2):
            self.done = False
            state = self.reset()
            print("start episode: ", episode)
            while not self.done:
                action = self.action_space.sample()
                state, reward, self.done, _ = self.step(action)
                print(
                    "step {}, action: {}, Current: {}, Best: {}, Reward: {:2.3f}".format(
                        self.iteration,
                        action,
                        self.current_solution.objective(),
                        self.best_solution.objective(),
                        reward,
                    )
                )
