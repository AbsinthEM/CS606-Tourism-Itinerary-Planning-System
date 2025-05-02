import argparse
from alns_main_stage1 import save_smjsp_output, create_tourist_summary, run_single_tourist
from stable_baselines3 import PPO

import os
from psp_AlnsEnv import SMJSPAlnsEnv
from rcjsp import get_objective_breakdown
from src.alns.Statistics import Statistics
from alns_main_stage1 import create_enhanced_summary

# model_path = ""  # // Include trained model path here (e.g. "src/dr_alns/trained_model/pspAlnsEnv/<IDX>_PPO_ActorCriticPolicy_<N_STEPS>_<N_WORKERS>_<DATE>/model.zip")
# model_path = "src/dr_alns/trained_models/pspAlnsEnv/39_PPO_ActorCriticPolicy_700000_10_04-04_16-51/model.zip" #13 10.33%
# model_path = "src/dr_alns/trained_models/pspAlnsEnv/40_PPO_ActorCriticPolicy_700000_10_04-04_20-17/model.zip" #13 11.88%
# model_path = "src/dr_alns/trained_models/pspAlnsEnv/42_PPO_ActorCriticPolicy_700000_10_04-04_21-02/model.zip" #16 #12.20%
# model_path = "src/dr_alns/trained_models/pspAlnsEnv/43_PPO_ActorCriticPolicy_700000_10_04-04_22-19/model.zip" #16 #12.35% 
model_path = "src/dr_alns/trained_models/pspAlnsEnv/44_PPO_ActorCriticPolicy_700000_10_04-04_23-16/model.zip" #16 #12.81%
# model_path = "src/dr_alns/trained_models/pspAlnsEnv/45_PPO_ActorCriticPolicy_700000_10_04-05_00-47/model.zip" #16 #12.18%
# model_path = "src/dr_alns/trained_models/pspAlnsEnv/46_PPO_ActorCriticPolicy_1000000_10_04-05_02-15/model.zip" #16 #12.44%
# model_path = "src/dr_alns/trained_models/pspAlnsEnv/49_PPO_ActorCriticPolicy_1000000_10_04-05_11-18/model.zip" #21 #10.79%
# model_path = "src/dr_alns/trained_models/pspAlnsEnv/52_PPO_ActorCriticPolicy_1000000_10_04-05_12-32/model.zip" #21 #6.9% - but change reward


if __name__ == "__main__":
    iterations = 1000  # // Modify number of ALNS iterations as you see fit

    model = PPO.load(model_path)
    
    parser = argparse.ArgumentParser(description='load data')
    # parser.add_argument(dest='tourist_id', type=str, help='Tourist ID')
    args = parser.parse_args()
    

    statistics = Statistics()

    seed = 606
    # tourist_id = int(args.tourist_id)
    
    objs = []

    # Save the aggregate results here
    total_initial_obj = 0.0
    total_optimized_obj = 0.0
    total_init_fun = 0.0
    total_init_distance = 0.0
    total_init_attractions = 0.0
    total_opt_fun = 0.0
    total_opt_distance = 0.0
    total_opt_attractions = 0.0
    count = 0

    # Enhanced data collection structures
    methods = ["initial", "optimized"]
    all_tourist_results = {}
    preference_groups = {}
    method_objectives = {"initial": [], "optimized": []}
    method_components = {
        "initial": {"fun": [], "distance": [], "attractions": [], "count": [], "money": []},
        "optimized": {"fun": [], "distance": [], "attractions": [], "count": [], "money": []}
    }
    
    for i in range(100):        
        parameters = {
            "environment": {
                "iterations": iterations,
                "tourist_id" : i + 1
            }
        }
        env = SMJSPAlnsEnv(parameters)
        env.run(model, seed = seed)

        init_solution = env.initial_solution
        init_objective = env.initial_objective
        init_pos_obj = - init_objective
        env.psp.update_included_categories()
        init_breakdown = get_objective_breakdown(env.psp)

        # Save init solutions
        init_solution.save_to_file(f"stage2_results/Tourist_{i}_initial_solution_DR_ALNS.json")

        # result
        solution = env.best_solution
        objective = env.best_solution.objective()
        objs.append(objective)
        print("Best objective is {}.".format(objective))

        best_obj_pos = -objective  # Convert to positive value
        solution.update_included_categories()
        best_breakdown = get_objective_breakdown(solution)
        solution.save_to_file(f"stage2_results/Tourist_{i}_optimized_solution_DR_ALNS.json")

        # Accumulate values for original summary
        total_initial_obj += init_pos_obj
        total_optimized_obj += best_obj_pos
        total_init_fun += init_breakdown['normalized_fun']
        total_init_distance += init_breakdown['normalized_distance']
        total_init_attractions += init_breakdown['normalized_attractions']
        total_opt_fun += best_breakdown['normalized_fun']
        total_opt_distance += best_breakdown['normalized_distance']
        total_opt_attractions += best_breakdown['normalized_attractions']

        # Collect enhanced metrics
        method_objectives["initial"].append(init_pos_obj)
        method_objectives["optimized"].append(best_obj_pos)
        
        method_components["initial"]["fun"].append(init_breakdown['normalized_fun'])
        method_components["initial"]["distance"].append(init_breakdown['normalized_distance'])
        method_components["initial"]["attractions"].append(init_breakdown['normalized_attractions'])
        method_components["initial"]["count"].append(init_breakdown['total_attractions'])
        method_components["initial"]["money"].append(init_solution.tourist.money_spent)
        
        method_components["optimized"]["fun"].append(best_breakdown['normalized_fun'])
        method_components["optimized"]["distance"].append(best_breakdown['normalized_distance'])
        method_components["optimized"]["attractions"].append(best_breakdown['normalized_attractions'])
        method_components["optimized"]["count"].append(best_breakdown['total_attractions'])
        method_components["optimized"]["money"].append(solution.tourist.money_spent)

        # Group by preference
        pref_key = '/'.join(sorted(init_solution.tourist.preferences))
        if pref_key not in preference_groups:
            preference_groups[pref_key] = {"initial": [], "optimized": []}
        preference_groups[pref_key]["initial"].append(init_pos_obj)
        preference_groups[pref_key]["optimized"].append(best_obj_pos)
        
        # Store full results for possible detailed analysis
        all_tourist_results[init_solution.tourist.idx] = {
            "initial": init_solution,
            "optimized": solution,
            "preference": pref_key
        }
        count += 1
        
        # generate output file
        # save_smjsp_output("SMJSP_DR_ALNS", solution, "solution" + str(seed))
        # create_tourist_summary(env.chosen_tourist.idx, env.initial_solution, solution, disrupt_names=env.disrupt_names)
        create_tourist_summary(env.chosen_tourist.idx, env.initial_solution, solution)
        
    # print(statistics.median(objs))

    # Compute average metrics across all tourists
    avg_initial_obj = total_initial_obj / count if count > 0 else 0
    avg_optimized_obj = total_optimized_obj / count if count > 0 else 0
    avg_init_fun = total_init_fun / count if count > 0 else 0
    avg_init_distance = total_init_distance / count if count > 0 else 0
    avg_init_attractions = total_init_attractions / count if count > 0 else 0
    avg_opt_fun = total_opt_fun / count if count > 0 else 0
    avg_opt_distance = total_opt_distance / count if count > 0 else 0
    avg_opt_attractions = total_opt_attractions / count if count > 0 else 0
    
    if avg_initial_obj >= 0 and avg_optimized_obj >= 0:
        # Both positive - standard percentage improvement
        improvement_pct = ((avg_optimized_obj - avg_initial_obj) / avg_initial_obj * 100) if avg_initial_obj > 0 else float('inf')
        improvement_desc = f"{improvement_pct:.2f}%"
    elif avg_initial_obj < 0 and avg_optimized_obj >= 0:
        # From negative to positive - complete reversal plus additional gain
        obj_improvement = abs(avg_initial_obj) + avg_optimized_obj
        improvement_desc = f"Complete reversal (positive gain of {obj_improvement:.4f})"
    elif avg_initial_obj < 0 and avg_optimized_obj < 0:
        # Both negative - calculate reduction in penalty
        obj_improvement = abs(avg_initial_obj) - abs(avg_optimized_obj)
        improvement_pct = (obj_improvement / abs(avg_initial_obj) * 100)
        improvement_desc = f"Penalty reduction of {improvement_pct:.2f}%"
    else:
        # Positive to negative - deterioration
        improvement_pct = ((avg_optimized_obj - avg_initial_obj) / avg_initial_obj * 100)
        improvement_desc = f"Deterioration of {abs(improvement_pct):.2f}%"

    summary = f"""
        Comparison Summary Across {count} Tourist Profiles:
        -----------------------------------------------------
        Average Initial Solution Objective (positive): {avg_initial_obj:.4f}
        Average Optimized Solution Objective (positive): {avg_optimized_obj:.4f}
        Overall Improvement: {improvement_desc}

        Average Normalized Components for Initial Solution:
        Fun:         {avg_init_fun:.4f}
        Distance:    {avg_init_distance:.4f}
        Attractions: {avg_init_attractions:.4f}

        Average Normalized Components for Optimized Solution:
        Fun:         {avg_opt_fun:.4f}
        Distance:    {avg_opt_distance:.4f}
        Attractions: {avg_opt_attractions:.4f}
        -----------------------------------------------------
        """
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_file = os.path.join(base_dir, "tourist_comparison_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary)
    
    # Create enhanced summary
    create_enhanced_summary(method_objectives, method_components, preference_groups, count)