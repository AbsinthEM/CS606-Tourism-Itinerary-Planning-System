"""
Stage 1: Initial planning and optimization without any disruptions.
"""

import os
import json
import argparse
import numpy.random as rnd
import sys
import random
import copy
import csv
from datetime import datetime
from rcjsp import SMJSP, Parser, get_objective_breakdown
from src.alns import ALNS
from src.alns.criteria import SimulatedAnnealing
# from disruptions import rainy_day_attraction, nothing_happens_attraction,  heat_wave_attraction, family_day, early_closure
from operators import (
    destroy_random, destroy_worst, destroy_day, destroy_preference, destroy_cluster,
    destroy_most_frequent_category, destroy_recently_added, destroy_expensive,
    repair_greedy, repair_regret, repair_random, repair_balanced, repair_nearest_neighbor,
    repair_maximize_attractions, repair_cheapest_first, repair_diversity_first, repair_time_slot_fit,
    repair_cheapest_fun_proximity, repair_rainy
)
from utils import calculate_distance
import math

# Set default weights globally
FUN_WEIGHT = 0.6
DISTANCE_WEIGHT = 0.3
ATTRACTION_WEIGHT = 0.5

# Normalize weights
w_list = [FUN_WEIGHT, DISTANCE_WEIGHT, ATTRACTION_WEIGHT]
s_ = sum(w_list)
w_norm = [v / s_ for v in w_list] if s_ > 0 else w_list



def save_smjsp_output_json(smjsp):
    # Build the output path
    base_dir     = os.path.dirname(os.path.abspath(__file__))
    react_public = os.path.join(base_dir, "dynamic-itinerary-planner", "public")
    filename     = os.path.join(react_public, "solution.json")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Build JSON structure
    solution = {
        "tourist": {
            "id": smjsp.tourist.idx,
            "budget": smjsp.tourist.budget,
            "hotelLat": smjsp.tourist.hotel_lat,
            "hotelLng": smjsp.tourist.hotel_long
        },
        "itinerary": []
    }

    for day in range(smjsp.tourist.days):
        day_entry = {"day": day + 1, "attractions": []}
        for attr in smjsp.tourist.locations.get(day, []):
            start = smjsp.tourist.start_times[day].get(attr.attraction_name)
            if start is None:
                continue
            end = start + attr.task_time
            day_entry["attractions"].append({
                "name": attr.attraction_name,
                "time": f"{start:.1f}-{end:.1f}",
                "cost": attr.cost,
                "categories": attr.categories,
                "lat": attr.lat_long[1] if len(attr.lat_long) >= 2 else None,
                "lng": attr.lat_long[0] if len(attr.lat_long) >= 2 else None
            })
        solution["itinerary"].append(day_entry)

    # Write JSON
    with open(filename, "w", encoding="utf-8") as fw:
        json.dump(solution, fw, indent=2)
    print(f"JSON solution saved to {filename}")

def save_smjsp_output(name, smjsp, suffix):
    """
    Save the tourism itinerary solution to a file (text format).
    Uses objective breakdown consistently throughout.
    """
    # Get all objective-related values from the breakdown
    bd = get_objective_breakdown(smjsp)
    
    # Create absolute path to ensure consistent location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(base_dir, f"{name}_{suffix}.txt")
    
    lines = []
    # Use values from breakdown dictionary consistently
    lines.append(f"Objective (negative): {bd['weighted_objective_negative']:.4f}")
    lines.append(f"Objective (positive): {bd['weighted_objective_positive']:.4f}")
    
    # Tourist details still come from the SMJSP object
    lines.append(f"Tourist ID: {smjsp.tourist.idx}")
    lines.append(f"Budget: SGD ${smjsp.tourist.budget}")
    lines.append(f"Money spent: SGD ${smjsp.tourist.money_spent}")
    lines.append(f"Preferences: {smjsp.tourist.preferences}")
    lines.append(f"Days: {smjsp.tourist.days}")
    lines.append(f"Unassigned attractions: {len(smjsp.unassigned)}")

    # Objective component breakdown from the dictionary
    lines.append("\n--- Objective Component Breakdown ---")
    lines.append(f"Fun Score (raw): {bd['fun_score_raw']:.2f}")
    lines.append(f"Distance (raw): {bd['distance_raw']:.2f} km")
    lines.append(f"Number of attractions: {bd['total_attractions']}")
    # lines.append(f"Empty days: {bd['empty_days']}")
    # lines.append(f"Empty-day penalty: {bd['empty_day_penalty']}")
    lines.append("")
    lines.append(f"Normalized Fun: {bd['normalized_fun']:.4f}")
    lines.append(f"Normalized Distance: {bd['normalized_distance']:.4f}")
    lines.append(f"Normalized #Attractions: {bd['normalized_attractions']:.4f}")
    lines.append("")
    lines.append(f"Weighted fun contribution: {bd['weighted_fun_contribution']:.4f}")
    lines.append(f"Weighted distance contribution: {bd['weighted_distance_contribution']:.4f}")
    lines.append(f"Weighted attractions contribution: {bd['weighted_attractions_contribution']:.4f}")
    lines.append("")
    lines.append(f"Weighted Objective (positive): {bd['weighted_objective_positive']:.4f}")

    # For each day
    for d in range(smjsp.tourist.days):
        lines.append(f"\nDay {d+1} forecast: Normal Day")

        if d in smjsp.tourist.locations and smjsp.tourist.locations[d]:
            acts = smjsp.tourist.locations[d]
            lines.append(f"Day {d+1} has {len(acts)} attractions:")

            # Sort by start time
            acts_times = []
            for a in acts:
                if a.attraction_name in smjsp.tourist.start_times[d]:
                    st_ = smjsp.tourist.start_times[d][a.attraction_name]
                    en_ = st_ + a.task_time
                    acts_times.append((a, st_, en_))
            acts_times.sort(key=lambda x: x[1])

            for idx_, (a, st, en) in enumerate(acts_times):
                lines.append(f"  - {a.attraction_name}")
                lines.append(f"    Time: {st:.1f} - {en:.1f}")
                lines.append(f"    Cost: SGD ${a.cost}")
                lines.append(f"    Categories: {a.categories}")

                # Travel
                if idx_ == 0:
                    prev_lat, prev_long = smjsp.tourist.hotel_lat, smjsp.tourist.hotel_long
                else:
                    prev_a, _, _ = acts_times[idx_ - 1]
                    prev_lat, prev_long = prev_a.lat_long[1], prev_a.lat_long[0]
                c_lat, c_long = a.lat_long[1], a.lat_long[0]
                dd = calculate_distance(prev_lat, prev_long, c_lat, c_long)
                ttime = dd / 30.0
                lines.append(f"    Travel: {dd:.2f} km (~{ttime:.2f} hours)")
                lines.append("")
        else:
            lines.append(f"Day {d+1}: No attractions")
    
    print(f"Saved output to: {filename}")
    
    




def save_comprehensive_evaluation(name, initial_solution, optimized_solution):
    """
    Comprehensive evaluation comparing initial, optimized in stage 1 without disruptions
    """
    # Get breakdowns
    initial_bd = get_objective_breakdown(initial_solution)
    optimized_bd = get_objective_breakdown(optimized_solution)
    
    # # Calculate improvements from initial to optimized
    # obj_improvement = (optimized_bd['weighted_objective_positive'] - initial_bd['weighted_objective_positive'])
    # obj_improvement_pct = (obj_improvement / initial_bd['weighted_objective_positive'] * 100) if initial_bd['weighted_objective_positive'] > 0 else 0
    
    # Calculate improvements from initial to optimized
    initial_obj = initial_bd['weighted_objective_positive']
    optimized_obj = optimized_bd['weighted_objective_positive']


    # Calculate improvement in a way that handles negative values properly

    if initial_obj >= 0 and optimized_obj >= 0:
        # Both positive - standard percentage improvement
        obj_improvement = optimized_obj - initial_obj
        obj_improvement_pct = (obj_improvement / initial_obj * 100) if initial_obj > 0 else float('inf')
        improvement_description = f"{obj_improvement_pct:.1f}%"

    elif initial_obj < 0 and optimized_obj >= 0:
        # From negative to positive - complete reversal plus additional gain
        obj_improvement = abs(initial_obj) + optimized_obj
        improvement_description = f"Complete reversal from {initial_obj:.4f} to {optimized_obj:.4f} (positive gain of {obj_improvement:.4f})"

    elif initial_obj < 0 and optimized_obj < 0:
        # Both negative - calculate reduction in penalty
        obj_improvement = abs(initial_obj) - abs(optimized_obj)
        obj_improvement_pct = (obj_improvement / abs(initial_obj) * 100) if initial_obj != 0 else 0
        improvement_description = f"Penalty reduction of {obj_improvement_pct:.1f}% ({initial_obj:.4f} to {optimized_obj:.4f})"

    else:
        # Positive to negative - deterioration
        obj_improvement = optimized_obj - initial_obj
        obj_improvement_pct = (obj_improvement / initial_obj * 100) if initial_obj != 0 else 0
        improvement_description = f"Deterioration of {abs(obj_improvement_pct):.1f}% ({initial_obj:.4f} to {optimized_obj:.4f})"
        
    # Create file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(base_dir, f"{name}_comprehensive_evaluation.txt")
    
    lines = []
    lines.append(f"COMPREHENSIVE ITINERARY PLANNING EVALUATION")
    lines.append(f"==========================================")
    lines.append(f"Tourist ID: {optimized_solution.tourist.idx}")
    lines.append(f"Preferences: {optimized_solution.tourist.preferences}")
    lines.append(f"Days: {optimized_solution.tourist.days}")
    lines.append("")
    
    # PART 1: Compare Initial vs Optimized Solution
    lines.append(f"INITIAL SOLUTION VS OPTIMIZED SOLUTION")
    lines.append(f"-------------------------------------")
    lines.append(f"{'Metric':<30} {'Initial':<15} {'Optimized':<15} {'Change':<15} {'% Change':<10}")
    lines.append(f"{'-'*75}")
    
    # Objective Value
    # lines.append(f"{'Objective Value':<30} {initial_bd['weighted_objective_positive']:.4f} {optimized_bd['weighted_objective_positive']:.4f} {obj_improvement:.4f} {obj_improvement_pct:+.1f}%")
    lines.append(f"Objective Value: {initial_bd['weighted_objective_positive']:.4f} → {optimized_bd['weighted_objective_positive']:.4f}")
    lines.append(f"Improvement: {improvement_description}")
    
    # # Component Breakdown - Raw Values
    # lines.append(f"\n{'RAW COMPONENT VALUES'}")
    # lines.append(f"{'Fun Score':<30} {initial_bd['fun_score_raw']:.1f} {optimized_bd['fun_score_raw']:.1f} {optimized_bd['fun_score_raw']-initial_bd['fun_score_raw']:.1f} {(optimized_bd['fun_score_raw']/initial_bd['fun_score_raw']-1)*100 if initial_bd['fun_score_raw'] > 0 else 0:+.1f}%")
    # lines.append(f"{'Distance (km)':<30} {initial_bd['distance_raw']:.1f} {optimized_bd['distance_raw']:.1f} {optimized_bd['distance_raw']-initial_bd['distance_raw']:.1f} {(optimized_bd['distance_raw']/initial_bd['distance_raw']-1)*100 if initial_bd['distance_raw'] > 0 else 0:+.1f}%")
    # lines.append(f"{'Number of Attractions':<30} {initial_bd['total_attractions']} {optimized_bd['total_attractions']} {optimized_bd['total_attractions']-initial_bd['total_attractions']} {(optimized_bd['total_attractions']/initial_bd['total_attractions']-1)*100 if initial_bd['total_attractions'] > 0 else 0:+.1f}%")
    
    # Component Breakdown - Normalized Values
    lines.append(f"\n{'NORMALIZED COMPONENTS'}")
    lines.append(f"{'Normalized Fun':<30} {initial_bd['normalized_fun']:.4f} {optimized_bd['normalized_fun']:.4f} {optimized_bd['normalized_fun']-initial_bd['normalized_fun']:.4f} {(optimized_bd['normalized_fun']/initial_bd['normalized_fun']-1)*100 if initial_bd['normalized_fun'] > 0 else 0:+.1f}%")
    lines.append(f"{'Normalized Distance':<30} {initial_bd['normalized_distance']:.4f} {optimized_bd['normalized_distance']:.4f} {optimized_bd['normalized_distance']-initial_bd['normalized_distance']:.4f} {(optimized_bd['normalized_distance']/initial_bd['normalized_distance']-1)*100 if initial_bd['normalized_distance'] > 0 else 0:+.1f}%")
    lines.append(f"{'Normalized Attractions':<30} {initial_bd['normalized_attractions']:.4f} {optimized_bd['normalized_attractions']:.4f} {optimized_bd['normalized_attractions']-initial_bd['normalized_attractions']:.4f} {(optimized_bd['normalized_attractions']/initial_bd['normalized_attractions']-1)*100 if initial_bd['normalized_attractions'] > 0 else 0:+.1f}%")
    
    # # Component Breakdown - Weighted Contributions
    # lines.append(f"\n{'WEIGHTED CONTRIBUTIONS'}")
    # lines.append(f"{'Fun Contribution':<30} {initial_bd['weighted_fun_contribution']:.4f} {optimized_bd['weighted_fun_contribution']:.4f} {optimized_bd['weighted_fun_contribution']-initial_bd['weighted_fun_contribution']:.4f} {(optimized_bd['weighted_fun_contribution']/initial_bd['weighted_fun_contribution']-1)*100 if initial_bd['weighted_fun_contribution'] > 0 else 0:+.1f}%")
    # lines.append(f"{'Distance Contribution':<30} {initial_bd['weighted_distance_contribution']:.4f} {optimized_bd['weighted_distance_contribution']:.4f} {optimized_bd['weighted_distance_contribution']-initial_bd['weighted_distance_contribution']:.4f} {(optimized_bd['weighted_distance_contribution']/initial_bd['weighted_distance_contribution']-1)*100 if initial_bd['weighted_distance_contribution'] > 0 else 0:+.1f}%")
    # lines.append(f"{'Attractions Contribution':<30} {initial_bd['weighted_attractions_contribution']:.4f} {optimized_bd['weighted_attractions_contribution']:.4f} {optimized_bd['weighted_attractions_contribution']-initial_bd['weighted_attractions_contribution']:.4f} {(optimized_bd['weighted_attractions_contribution']/initial_bd['weighted_attractions_contribution']-1)*100 if initial_bd['weighted_attractions_contribution'] > 0 else 0:+.1f}%")
    

            
# Define a function to run single tourist generation, optimization, and evaluation
def solve_single_tourist(tourist, attractions, seed=123):
    """
    Generates initial and optimized solutions for a single tourist, compares them,
    and saves output files.
    """
    
    print(f"\n--- Generating itinerary for Tourist ID: {tourist.idx} ---")

    # Build SMJSP
    smjsp = SMJSP(
        tourist,
        attractions,
        weighting=w_norm, # pass normalized wights
        diversity_bonus=15
    )

    # Initial
    print("\nConstructing initial solution ...")
    init_neg_obj = smjsp.random_initialize(seed)
    smjsp.update_included_categories()
    init_pos_obj = -init_neg_obj
    print(f"Initial objective(negative): {init_neg_obj:.4f}")
    print(f"Initial objective(positive): {init_pos_obj:.4f}")

    # Save initial solution
    save_smjsp_output(f"Tourist_{tourist.idx}_Initial_Itinerary", smjsp, "initial")
    
    # Keep a copy of the initial solution
    initial_solution = smjsp.copy()
    
    # Save serialized initial solution
    initial_solution.save_to_file(f"Tourist_{tourist.idx}_initial_solution.json")

    # ALNS
    random_state = rnd.RandomState(seed)
    alns = ALNS(random_state)

    # Add operators
    alns.add_destroy_operator(destroy_random, "Random")
    alns.add_destroy_operator(destroy_worst, "Worst")
    alns.add_destroy_operator(destroy_day, "Day")
    alns.add_destroy_operator(destroy_preference, "Preference")
    alns.add_destroy_operator(destroy_cluster, "Cluster")
    # alns.add_destroy_operator(destroy_most_frequent_category, "MostFreq")
    alns.add_destroy_operator(destroy_recently_added, "Recent")
    alns.add_destroy_operator(destroy_expensive, "Expensive")

    alns.add_repair_operator(repair_random, "Random")
    alns.add_repair_operator(repair_greedy, "Greedy")
    alns.add_repair_operator(repair_regret, "Regret")
    # alns.add_repair_operator(repair_balanced, "Balanced")
    alns.add_repair_operator(repair_nearest_neighbor, "Nearest")
    alns.add_repair_operator(repair_rainy, "Rainy")
    # alns.add_repair_operator(repair_cheapest_first, "CheapestFirst")
    alns.add_repair_operator(repair_diversity_first, "DiversityFirst")
    alns.add_repair_operator(repair_time_slot_fit, "TimeSlotFit")
    alns.add_repair_operator(repair_cheapest_fun_proximity, "CheapestFunProx")
    alns.add_repair_operator(repair_maximize_attractions, "MaxAttr")

    # ALNS params
    omegas = [5, 3, 2, 1]
    lambda_ = 0.8
    criterion = SimulatedAnnealing(200, 0.5, 0.9999, "exponential")

    # Run ALNS
    print("\nRunning ALNS optimization ...")
    result = alns.iterate(
        smjsp,
        omegas,
        lambda_,
        criterion,
        iterations=1000,
        collect_stats=True
    )

    best_sol = result.best_state
    
    final_neg = best_sol.objective()
    final_pos = -final_neg
    print(f"\nFinal solution objective (negative): {final_neg:.4f}")
    print(f"Final solution objective (positive): {final_pos:.4f}")

    best_sol.update_included_categories()
    
    method_objectives = {"initial": [], "optimized": []}
    method_components = {

        "initial": {"fun": [], "distance": [], "attractions": [], "count": [], "money": []},
        "optimized": {"fun": [], "distance": [], "attractions": [], "count": [], "money": []}
    }
    preference_groups = {}
    count = 1  # Just handling one tourist
    
    # Add data for this tourist
    method_objectives["initial"].append(init_pos_obj)
    method_objectives["optimized"].append(final_pos)

    
    # Add component data (get these from the breakdown)
    initial_breakdown = get_objective_breakdown(initial_solution)
    optimized_breakdown = get_objective_breakdown(best_sol)
    
    method_components["initial"]["fun"].append(initial_breakdown["normalized_fun"])
    method_components["initial"]["distance"].append(initial_breakdown["normalized_distance"])
    method_components["initial"]["attractions"].append(initial_breakdown["normalized_attractions"])
    method_components["initial"]["count"].append(initial_breakdown["total_attractions"])
    method_components["initial"]["money"].append(initial_solution.tourist.money_spent)
    method_components["optimized"]["fun"].append(optimized_breakdown["normalized_fun"])
    method_components["optimized"]["distance"].append(optimized_breakdown["normalized_distance"])
    method_components["optimized"]["attractions"].append(optimized_breakdown["normalized_attractions"])
    method_components["optimized"]["count"].append(optimized_breakdown["total_attractions"])
    method_components["optimized"]["money"].append(best_sol.tourist.money_spent)
    
    # Setup preference group
    pref_key = '/'.join(sorted(tourist.preferences))
    preference_groups[pref_key] = {"initial": [init_pos_obj], "optimized": [final_pos]}
    
    create_tourist_summary(tourist.idx, initial_solution, best_sol)
    create_enhanced_summary(method_objectives, method_components, preference_groups, count)
    
    # # Evaluation
    # disruption_results = evaluate_dynamic_replanning(best_sol, num_scenarios=5, seed=seed+100)

    # Save final solution
    save_smjsp_output(f"Tourist_{tourist.idx}_Final_Optimized_Itinerary", best_sol, "solution")
    
    # Save serialized optimized solution
    best_sol.save_to_file(f"Tourist_{tourist.idx}_optimized_solution.json")

    # Save comprehensive evaluation for stage 1
    save_comprehensive_evaluation(f"Tourist_{tourist.idx}_Dynamic_Tour_Planner", initial_solution, best_sol)
    save_smjsp_output_json(best_sol)
    return {"initial":initial_solution, "optimised":best_sol}



def aggregate_results(attraction_csv, tourist_csv, seed=123):
    """
    Loops through all tourist profiles and for each:
    - Constructs an initial solution via random_initialize,
    - Optimizes the solution using ALNS,
    - Retrieves the positive objective value and the normalized breakdown
    - Saves both initial and optimized solutions to files for later use in Stage 2
    - Generates both standard and enhanced aggregate summaries
    """

    # Parse the input data
    parser = Parser(attraction_csv, tourist_csv)
    
    # Adjust attraction opening hours keys (assume original keys are 1-indexed)
    for attraction in parser.attractions:
        new_opening = {}
        for day, hours in attraction.opening_hours.items():
            new_opening[day - 1] = hours
        attraction.opening_hours = new_opening

    # Accumulators for aggregated statistics
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

    # Loop over each tourist
    for tourist in parser.tourists:
        print(f"Processing Tourist ID: {tourist.idx} ({count+1}/{len(parser.tourists)})")
        
        # Create initial solution for the current tourist
        smjsp_initial = SMJSP(
            tourist,
            parser.attractions,
            weighting=w_norm, # pass normalized weights
            diversity_bonus=15
        )
        init_neg_obj = smjsp_initial.random_initialize(seed)
        init_pos_obj = -init_neg_obj  # Convert to positive objective value
        smjsp_initial.update_included_categories()
        init_breakdown = get_objective_breakdown(smjsp_initial)
        
        # Save the initial solution to file
        initial_solution = smjsp_initial.copy()
        initial_solution.save_to_file(f"Tourist_{tourist.idx}_initial_solution.json")
        
        # Set up ALNS optimization using the same initial solution
        random_state = rnd.RandomState(seed)
        alns = ALNS(random_state)
        
        # Add operators
        alns.add_destroy_operator(destroy_random, "Random")
        alns.add_destroy_operator(destroy_worst, "Worst")
        alns.add_destroy_operator(destroy_day, "Day")
        alns.add_destroy_operator(destroy_preference, "Preference")
        alns.add_destroy_operator(destroy_cluster, "Cluster")
        alns.add_destroy_operator(destroy_recently_added, "Recent")
        alns.add_destroy_operator(destroy_expensive, "Expensive")

        alns.add_repair_operator(repair_random, "Random")
        alns.add_repair_operator(repair_greedy, "Greedy")
        alns.add_repair_operator(repair_regret, "Regret")
        alns.add_repair_operator(repair_nearest_neighbor, "Nearest")
        alns.add_repair_operator(repair_rainy, "Rainy")
        alns.add_repair_operator(repair_diversity_first, "DiversityFirst")
        alns.add_repair_operator(repair_time_slot_fit, "TimeSlotFit")
        alns.add_repair_operator(repair_cheapest_fun_proximity, "CheapestFunProx")
        alns.add_repair_operator(repair_maximize_attractions, "MaxAttr")
        
        omegas = [5, 3, 2, 1]
        lambda_ = 0.8
        criterion = SimulatedAnnealing(200, 0.5, 0.9999, "exponential")
        
        result = alns.iterate(
            smjsp_initial,
            omegas,
            lambda_,
            criterion,
            iterations=1000,
            collect_stats=True
        )
        best_sol = result.best_state
        best_obj_neg = best_sol.objective()
        best_obj_pos = -best_obj_neg  # Convert to positive value
        best_sol.update_included_categories()
        best_breakdown = get_objective_breakdown(best_sol)
        
        # Save the optimized solution to file
        best_sol.save_to_file(f"Tourist_{tourist.idx}_optimized_solution.json")

        #Save to public JSON for UI
        save_smjsp_output_json(best_sol)
        
        # Create an individual tourist summary
        create_tourist_summary(tourist.idx, initial_solution, best_sol)
        
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
        method_components["initial"]["money"].append(initial_solution.tourist.money_spent)
        
        method_components["optimized"]["fun"].append(best_breakdown['normalized_fun'])
        method_components["optimized"]["distance"].append(best_breakdown['normalized_distance'])
        method_components["optimized"]["attractions"].append(best_breakdown['normalized_attractions'])
        method_components["optimized"]["count"].append(best_breakdown['total_attractions'])
        method_components["optimized"]["money"].append(best_sol.tourist.money_spent)
        
        # Group by preference
        pref_key = '/'.join(sorted(tourist.preferences))
        if pref_key not in preference_groups:
            preference_groups[pref_key] = {"initial": [], "optimized": []}
        preference_groups[pref_key]["initial"].append(init_pos_obj)
        preference_groups[pref_key]["optimized"].append(best_obj_pos)
        
        # Store full results for possible detailed analysis
        all_tourist_results[tourist.idx] = {
            "initial": initial_solution,
            "optimized": best_sol,
            "preference": pref_key
        }
        
        count += 1

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

    # Build the original summary string
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
    # Print summary to console
    print(summary)
    
    # Save original summary to file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_file = os.path.join(base_dir, "tourist_comparison_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary)
    
    # Create enhanced summary
    create_enhanced_summary(method_objectives, method_components, preference_groups, count)
    
    print(f"Summary saved to: {summary_file}")
    print(f"Enhanced summary saved to: enhanced_tourist_comparison_summary.txt")
    print(f"Saved {count} initial and optimized solutions for use in Stage 2.")

def create_tourist_summary(tourist_id, initial_solution, optimized_solution):
    """Create a comprehensive summary for an individual tourist."""
    initial_bd = get_objective_breakdown(initial_solution)
    optimized_bd = get_objective_breakdown(optimized_solution)
    
    # Calculate improvement metrics
    initial_obj = initial_bd['weighted_objective_positive']
    optimized_obj = optimized_bd['weighted_objective_positive']
    
    # Create improvement description text using the same logic as Stage 2
    if initial_obj >= 0 and optimized_obj >= 0:
        obj_improvement = optimized_obj - initial_obj
        obj_improvement_pct = (obj_improvement / initial_obj * 100) if initial_obj > 0 else float('inf')
        improvement_description = f"{obj_improvement_pct:.1f}%"
    elif initial_obj < 0 and optimized_obj >= 0:
        obj_improvement = abs(initial_obj) + optimized_obj
        improvement_description = f"Complete reversal from {initial_obj:.4f} to {optimized_obj:.4f} (positive gain of {obj_improvement:.4f})"
    elif initial_obj < 0 and optimized_obj < 0:
        obj_improvement = abs(initial_obj) - abs(optimized_obj)
        obj_improvement_pct = (obj_improvement / abs(initial_obj) * 100) if initial_obj != 0 else 0
        improvement_description = f"Penalty reduction of {obj_improvement_pct:.1f}% ({initial_obj:.4f} to {optimized_obj:.4f})"
    else:
        obj_improvement = optimized_obj - initial_obj
        obj_improvement_pct = (obj_improvement / initial_obj * 100) if initial_obj != 0 else 0
        improvement_description = f"Deterioration of {abs(obj_improvement_pct):.1f}% ({initial_obj:.4f} to {optimized_obj:.4f})"
    
    # Create output file
    filename = f"Tourist_{tourist_id}_Summary.txt"
    with open(filename, 'w') as f:
        # Header
        f.write(f"TOURIST ITINERARY PLANNING SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Tourist ID: {tourist_id}\n")
        f.write(f"Preferences: {initial_solution.tourist.preferences}\n")
        f.write(f"Days: {initial_solution.tourist.days}\n")
        f.write(f"Budget: SGD ${initial_solution.tourist.budget}\n\n")
        
        # Objective Values
        f.write("OBJECTIVE VALUES\n")
        f.write("-" * 70 + "\n")
        f.write(f"Initial Solution: {initial_obj:.4f}\n")
        f.write(f"Optimized Solution: {optimized_obj:.4f}\n")
        f.write(f"Improvement: {improvement_description}\n\n")
        
        # Component Breakdown
        f.write("NORMALIZED COMPONENT VALUES\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Component':<20} {'Initial':<15} {'Optimized':<15} {'Change (%)':<10}\n")
        
        # Calculate component changes
        fun_change = ((optimized_bd['normalized_fun'] / initial_bd['normalized_fun']) - 1) * 100 if initial_bd['normalized_fun'] > 0 else float('inf')
        dist_change = ((optimized_bd['normalized_distance'] / initial_bd['normalized_distance']) - 1) * 100 if initial_bd['normalized_distance'] > 0 else float('inf')
        attr_change = ((optimized_bd['normalized_attractions'] / initial_bd['normalized_attractions']) - 1) * 100 if initial_bd['normalized_attractions'] > 0 else float('inf')
        
        f.write(f"{'Fun Score':<20} {initial_bd['normalized_fun']:<15.4f} {optimized_bd['normalized_fun']:<15.4f} {fun_change:+.1f}%\n")
        f.write(f"{'Distance':<20} {initial_bd['normalized_distance']:<15.4f} {optimized_bd['normalized_distance']:<15.4f} {dist_change:+.1f}%\n")
        f.write(f"{'Attractions':<20} {initial_bd['normalized_attractions']:<15.4f} {optimized_bd['normalized_attractions']:<15.4f} {attr_change:+.1f}%\n\n")
        
        # Raw metrics
        f.write("RAW METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Metric':<20} {'Initial':<15} {'Optimized':<15}\n")
        f.write(f"{'Attractions Count':<20} {initial_bd['total_attractions']:<15d} {optimized_bd['total_attractions']:<15d}\n")
        f.write(f"{'Total Distance (km)':<20} {initial_bd['distance_raw']:<15.2f} {optimized_bd['distance_raw']:<15.2f}\n")
        f.write(f"{'Raw Fun Score':<20} {initial_bd['fun_score_raw']:<15.2f} {optimized_bd['fun_score_raw']:<15.2f}\n")
        f.write(f"{'Money Spent ($)':<20} {initial_solution.tourist.money_spent:<15d} {optimized_solution.tourist.money_spent:<15d}\n\n")
        
        # Itinerary Summary
        f.write("ITINERARY SUMMARY\n")
        f.write("-" * 70 + "\n")
        
        # Initial solution
        f.write("Initial Solution:\n")
        for day in range(initial_solution.tourist.days):
            attractions = initial_solution.tourist.locations.get(day, [])
            if attractions:
                f.write(f"  Day {day+1}: {len(attractions)} attractions - {', '.join(a.attraction_name for a in attractions)}\n")
            else:
                f.write(f"  Day {day+1}: No attractions\n")
        
        f.write("\nOptimized Solution:\n")
        for day in range(optimized_solution.tourist.days):
            attractions = optimized_solution.tourist.locations.get(day, [])
            if attractions:
                f.write(f"  Day {day+1}: {len(attractions)} attractions - {', '.join(a.attraction_name for a in attractions)}\n")
            else:
                f.write(f"  Day {day+1}: No attractions\n")
        
        # Added categories
        f.write("\nCATEGORIES INCLUDED\n")
        f.write("-" * 70 + "\n")
        initial_categories = set(initial_solution.included_categories)
        optimized_categories = set(optimized_solution.included_categories)
        new_categories = optimized_categories - initial_categories
        
        f.write(f"Initial Solution: {', '.join(sorted(initial_categories))}\n")
        f.write(f"Optimized Solution: {', '.join(sorted(optimized_categories))}\n")
        if new_categories:
            f.write(f"New Categories Added: {', '.join(sorted(new_categories))}\n")
    
    print(f"Created summary for Tourist {tourist_id}: {filename}")
    return filename

def create_enhanced_summary(method_objectives, method_components, preference_groups, count):
    """Create enhanced aggregate summary for Stage 1 results."""
    filename = "enhanced_tourist_comparison_summary.txt"
    
    with open(filename, 'w') as f:
        f.write("ENHANCED TOURIST ITINERARY PLANNING SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        # Overall Method Performance
        f.write("OVERALL METHOD PERFORMANCE\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Method':<15} {'Avg Objective':<15} {'Min':<10} {'Max':<10} {'Std Dev':<10}\n")
        f.write("-" * 70 + "\n")
        
        for method in ["initial", "optimized"]:
            values = method_objectives[method]
            if values:
                avg = sum(values) / len(values)
                min_val = min(values)
                max_val = max(values)
                std_dev = (sum((x - avg) ** 2 for x in values) / len(values)) ** 0.5 if len(values) > 1 else 0
                f.write(f"{method:<15} {avg:<15.4f} {min_val:<10.4f} {max_val:<10.4f} {std_dev:<10.4f}\n")
            else:
                f.write(f"{method:<15} No data\n")
        
        # Overall improvement
        if method_objectives["initial"] and method_objectives["optimized"]:
            avg_initial = sum(method_objectives["initial"]) / len(method_objectives["initial"])
            avg_optimized = sum(method_objectives["optimized"]) / len(method_objectives["optimized"])
            
            # Calculate improvement description
            if avg_initial >= 0 and avg_optimized >= 0:
                improvement_pct = ((avg_optimized - avg_initial) / avg_initial * 100) if avg_initial > 0 else float('inf')
                improvement_desc = f"{improvement_pct:.2f}%"
            elif avg_initial < 0 and avg_optimized >= 0:
                obj_improvement = abs(avg_initial) + avg_optimized
                improvement_desc = f"Complete reversal (positive gain of {obj_improvement:.4f})"
            elif avg_initial < 0 and avg_optimized < 0:
                obj_improvement = abs(avg_initial) - abs(avg_optimized)
                improvement_pct = (obj_improvement / abs(avg_initial) * 100)
                improvement_desc = f"Penalty reduction of {improvement_pct:.2f}%"
            else:
                improvement_pct = ((avg_optimized - avg_initial) / avg_initial * 100)
                improvement_desc = f"Deterioration of {abs(improvement_pct):.2f}%"
            
            f.write(f"\nOverall Improvement: {improvement_desc}\n\n")
            
        # Performance by Tourist Preference
        f.write("\nPERFORMANCE BY TOURIST PREFERENCE\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Preference':<25} {'Initial':<15} {'Optimized':<15} {'Improvement':<15}\n")
        f.write("-" * 70 + "\n")
        
        for pref, pref_data in preference_groups.items():
            if pref_data["initial"] and pref_data["optimized"]:
                pref_initial = sum(pref_data["initial"]) / len(pref_data["initial"])
                pref_optimized = sum(pref_data["optimized"]) / len(pref_data["optimized"])
                
                # Calculate improvement for this preference group
                if pref_initial < 0 and pref_optimized < 0:
                    pref_improvement = f"{(abs(pref_initial) - abs(pref_optimized)) / abs(pref_initial) * 100:.1f}% reduced"
                elif pref_initial < 0 and pref_optimized >= 0:
                    pref_improvement = "Complete reversal"
                else:
                    pref_improvement = f"{(pref_optimized - pref_initial) / abs(pref_initial) * 100:.1f}%"
                
                f.write(f"{pref:<25} {pref_initial:<15.4f} {pref_optimized:<15.4f} {pref_improvement:<15}\n")
        
        # Normalized Components Analysis
        f.write("\nNORMALIZED COMPONENT ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Component':<15} {'Initial Avg':<12} {'Optimized Avg':<15} {'Change':<10}\n")
        f.write("-" * 70 + "\n")
        
        # Calculate component averages
        for component in ["fun", "distance", "attractions"]:
            initial_avg = sum(method_components["initial"][component]) / len(method_components["initial"][component])
            opt_avg = sum(method_components["optimized"][component]) / len(method_components["optimized"][component])
            change_pct = ((opt_avg / initial_avg) - 1) * 100 if initial_avg > 0 else float('inf')
            
            f.write(f"{component.capitalize():<15} {initial_avg:<12.4f} {opt_avg:<15.4f} {change_pct:+.1f}%\n")
        
        # Attraction Count Stats
        initial_count_avg = sum(method_components["initial"]["count"]) / len(method_components["initial"]["count"])
        opt_count_avg = sum(method_components["optimized"]["count"]) / len(method_components["optimized"]["count"])
        count_change = ((opt_count_avg / initial_count_avg) - 1) * 100 if initial_count_avg > 0 else float('inf')
        
        f.write(f"\nAverage Attractions Per Tourist:\n")
        f.write(f"Initial: {initial_count_avg:.1f}, Optimized: {opt_count_avg:.1f} ({count_change:+.1f}%)\n")
        
        # Money spent
        initial_money_avg = sum(method_components["initial"]["money"]) / len(method_components["initial"]["money"])
        opt_money_avg = sum(method_components["optimized"]["money"]) / len(method_components["optimized"]["money"])
        money_change = ((opt_money_avg / initial_money_avg) - 1) * 100 if initial_money_avg > 0 else float('inf')
        
        f.write(f"\nAverage Money Spent Per Tourist:\n")
        f.write(f"Initial: ${initial_money_avg:.1f}, Optimized: ${opt_money_avg:.1f} ({money_change:+.1f}%)\n")
        
        f.write("\n")
        f.write(f"Summary generated for {count} tourists on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Enhanced aggregate summary saved to: {filename}")
    return filename

def run_single_tourist(attraction_csv, tourist_csv, seed, tourist_id=None):
    """
    Handles the case where a specific tourist ID is provided. Generates a plan,
    optimizes it, and saves the output.
    """
    # Parse data
    parsed = Parser(attraction_csv, tourist_csv)

    # Shift day indexing
    for a_ in parsed.attractions:
        new_opening = {}
        for dayk, hourset in a_.opening_hours.items():
            new_opening[dayk-1] = hourset
        a_.opening_hours = new_opening

    # Choose touristS
    if tourist_id is not None:
        chosen = None
        for t_ in parsed.tourists:
            if int(t_.idx) == tourist_id:
                chosen = t_
                break
        if not chosen:
            print(f"Tourist ID {tourist_id} not found, abort.")
            sys.exit(1)
    else:
        # If no tourist ID is supplied, randomly select one
        chosen = random.choice(parsed.tourists)

    print(f"\nPlanning itinerary for Tourist ID: {chosen.idx}")

    # Generate, Optimize, and Evaluate (Single)
    solution = solve_single_tourist(chosen, parsed.attractions, seed)
    
    create_tourist_summary(chosen.idx, solution["initial"], solution["optimised"])

    print("\nItinerary planning complete!")
    

def run_aggregate_results(attraction_csv, tourist_csv, seed):
    # Perform aggregated analysis across all tourists (Aggregate)
    aggregate_results(attraction_csv, tourist_csv, seed)
    
    all_results = {}

    for tourist in parsed.tourists:
        initial_path = f"Tourist_{tourist.idx}_initial_solution.json"
        optimized_path = f"Tourist_{tourist.idx}_optimized_solution.json"
        

        if os.path.exists(initial_path) and os.path.exists(optimized_path):
            initial_sol = SMJSP.load_from_file(initial_path, parsed.attractions)
            optimized_sol = SMJSP.load_from_file(optimized_path, parsed.attractions)
            all_results[tourist.idx] = {"initial": initial_sol, "optimized": optimized_sol}
    
    create_enhanced_summary(all_results, "results")

from rcjsp import Tourist, Parser

def run_custom_tourist_from_input(params, seed=123):
    """
    Accepts user-defined params from the React frontend and runs ALNS.
    Returns the optimized SMJSP solution object.
    """
    attraction_csv = os.path.join(os.path.dirname(__file__), "AttractionProfile.csv")
    tourist_csv = os.path.join(os.path.dirname(__file__), "TouristProfile.csv")

    # Construct tourist object manually from params
    tourist_data = [
        str(params.get("id", "999")),  # fallback to dummy ID
        str(params.get("preferences", [])),
        str(params.get("budget", 0)),
        str(params.get("days", 1)),
        str(params.get("touring_hours", [8, 19])),
        str(params.get("hotelLong", 0)),
        str(params.get("hotelLat", 0))
    ]
    tourist = Tourist(tourist_data)

    # Load attractions and adjust day keys
    parsed = Parser(attraction_csv, tourist_csv)
    for a_ in parsed.attractions:
        new_opening = {}
        for dayk, hourset in a_.opening_hours.items():
            new_opening[dayk - 1] = hourset
        a_.opening_hours = new_opening

    # Solve for this single tourist
    result = solve_single_tourist(tourist, parsed.attractions, seed)
    return result["optimised"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tourist Itinerary Planning using ALNS")
    parser.add_argument("attraction_data", type=str, help="attraction data CSV file")
    parser.add_argument("tourist_data", type=str, help="tourist data CSV file")
    parser.add_argument("seed", type=int, help="random seed")
    parser.add_argument("--tourist_id", type=int, default=None, help="specific tourist ID to plan for")
    parser.add_argument(
        "--mode", type=str, default="aggregate", choices=["aggregate", "single"],
        help="Mode of operation: 'aggregate' for all tourists, 'single' for a specific tourist."
    )
    parser.add_argument("--diversity_bonus", type=float, default=15, help="bonus points for first attraction of each category")
    parser.add_argument("--fun_weight", type=float, default=0.6, help="weight for fun score in objective function") # will be normalized later 
    parser.add_argument("--distance_weight", type=float, default=0.3, help="weight for distance in objective function")  # Decreased from 0.4, will be normalized later 
    parser.add_argument("--attraction_weight", type=float, default=0.5, help="weight for number of attractions in objective function")  # Increased from 0.3, will be normalized later 
    args = parser.parse_args()

    if args.mode == "single":
        run_single_tourist(args.attraction_data, args.tourist_data, args.seed, args.tourist_id)
    elif args.mode == "aggregate":
        # Perform aggregated analysis across all tourists (Aggregate)
        aggregate_results(args.attraction_data, args.tourist_data, args.seed)
    else:
        print("Invalid mode. Please use 'aggregate' or 'single'.")

def run_aggregate_results(attraction_csv, tourist_csv, seed):
    # Perform aggregated analysis across all tourists (Aggregate)
    aggregate_results(attraction_csv, tourist_csv, seed)