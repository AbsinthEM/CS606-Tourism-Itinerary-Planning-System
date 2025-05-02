"""
Stage 1: Initial planning and optimization without any disruptions.
"""

import argparse
import os
import sys
from datetime import datetime

import numpy.random as rnd
from operators import *
from rcjsp import Parser, get_objective_breakdown
from src.alns import ALNS
from src.alns.criteria import SimulatedAnnealing
from utils import calculate_distance
from evaluation import calculate_objective_metrics, compare_solutions

# Create directory for stage1 results
def ensure_results_directory():
    """
    Ensures that the stage1_results directory exists. Creates it if it doesn't.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "stage1_results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def save_smjsp_output(name, smjsp, suffix):
    """
    Save the tourism itinerary solution to a file (text format).
    Uses objective breakdown consistently throughout.
    """
    # Get all objective-related values from the breakdown
    bd = get_objective_breakdown(smjsp)

    # Create absolute path to ensure consistent location in stage1_results directory
    results_dir = ensure_results_directory()
    filename = os.path.join(results_dir, f"{name}_{suffix}.txt")

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
        lines.append(f"\nDay {d + 1} forecast: Normal Day")

        if d in smjsp.tourist.locations and smjsp.tourist.locations[d]:
            acts = smjsp.tourist.locations[d]
            lines.append(f"Day {d + 1} has {len(acts)} attractions:")

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
            lines.append(f"Day {d + 1}: No attractions")

    # Build the output path for web/React
    base_dir = os.path.dirname(os.path.abspath(__file__))
    react_public = os.path.join(base_dir, "dynamic-itinerary-planner", "public")
    json_filename = os.path.join(react_public, "solution.json")
    os.makedirs(os.path.dirname(json_filename), exist_ok=True)

    # Build JSON structure for web interface
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

    # Write the text output to stage1_results directory
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Write JSON for web interface
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(solution, f, indent=2)

    print(f"Saved output to: {filename}")
    print(f"Saved JSON to: {json_filename}")


# Define a function to run single tourist generation, optimization, and evaluation
def solve_single_tourist(tourist, attractions, seed=123):
    """
    Generates initial and optimized solutions for a single tourist, compares them,
    and saves output files. Ensures optimized solution is never worse than initial
    and validates all constraints before finalizing.
    """

    print(f"\n--- Generating itinerary for Tourist ID: {tourist.idx} ---")

    # Build SMJSP
    smjsp = SMJSP(
        tourist,
        attractions,
        weighting=w_norm,  # pass normalized weights
        diversity_bonus=15
    )

    # Initial
    print("\nConstructing initial solution ...")
    init_neg_obj = smjsp.random_initialize(seed)
    smjsp.update_included_categories()
    
    init_pos_obj = -init_neg_obj
    print(f"Initial objective(negative): {init_neg_obj:.4f}")
    print(f"Initial objective(positive): {init_pos_obj:.4f}")

    # Keep a copy of the initial solution
    initial_solution = smjsp.copy()
    
    # Get detailed breakdown for metrics
    _, breakdown = initial_solution.objective(return_breakdown=True)
    
    # Store basic metrics for later comparison
    initial_metrics = {
    "objective_negative": init_neg_obj,
    "objective_positive": init_pos_obj,
    "fun_score": breakdown["fun_score_raw"],
    "distance": breakdown["distance_raw"],
    "attractions_count": sum(len(initial_solution.tourist.locations.get(d, [])) for d in range(initial_solution.tourist.days)),
    "money_spent": initial_solution.tourist.money_spent,
    "normalized_fun": breakdown["normalized_fun"],
    "normalized_distance": breakdown["normalized_distance"],
    "normalized_attractions": breakdown["normalized_attractions"],
    "breakdown": breakdown
}
    # Always save initial solution JSON (needed for Stage 2)
    results_dir = ensure_results_directory()
    initial_json_path = os.path.join(results_dir, f"Tourist_{tourist.idx}_initial_solution.json")
    initial_solution.save_to_file(initial_json_path)

    # Optionally save initial solution text output
    if args.save_initial:
        save_smjsp_output(f"Tourist_{tourist.idx}_Initial_Itinerary", initial_solution, "initial")

    # ALNS
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
    

    # ALNS params - Using POSITIONAL arguments only
    omegas = [7, 5, 3, 1]  # Increased weights for successful operators
    lambda_ = 0.8

    # Fixed SimulatedAnnealing parameters using positional arguments only
    # (start_temp, end_temp, decay_rate, method)
    criterion = SimulatedAnnealing(300, 0.1, 0.9995, "exponential")

    # Run multiple trials of ALNS and keep the best result
    best_neg_objective = float('inf')  # Start with worst possible value (high negative objective)
    best_sol = None
    num_trials = 3  # Run multiple optimization trials

    print("\nRunning ALNS optimization with multiple trials...")
    for trial in range(num_trials):
        print(f"Trial {trial + 1}/{num_trials}...")

        # Start each trial with the initial solution
        trial_smjsp = initial_solution.copy()

        # Run ALNS
        result = alns.iterate(
            trial_smjsp,
            omegas,
            lambda_,
            criterion,
            iterations=5000,
            collect_stats=True
        )

        trial_solution = result.best_state
        trial_neg_obj = trial_solution.objective()
        trial_pos_obj = -trial_neg_obj

        print(f"  Trial {trial + 1} objective: {trial_neg_obj:.4f}")

        # Update best solution if this trial yielded better results (lower negative objective)
        if trial_neg_obj < best_neg_objective:
            best_neg_objective = trial_neg_obj
            best_sol = trial_solution

    # Critical improvement: Make sure optimized is never worse than initial
    initial_neg_obj = initial_solution.objective()
    final_neg_obj = best_sol.objective()

    # For negative objectives, lower is better, so if optimized is higher than initial, it's worse
    if final_neg_obj > initial_neg_obj:
        print(f"\nWARNING: Optimization did not improve solution")
        print(f"Initial objective (negative): {initial_neg_obj:.4f}, optimized: {final_neg_obj:.4f}")
        print("Reverting to initial solution")
        best_sol = initial_solution.copy()
        final_neg_obj = initial_neg_obj

    # POST-OPTIMIZATION CONSTRAINT VALIDATION AND FIXING
    print("\nValidating and fixing constraints...")
    constraints_valid = False
    max_iterations = 3  # Maximum iterations to attempt fixing

    for iteration in range(max_iterations):
        # Assume valid until proven otherwise
        constraints_valid = True

        # 1. VALIDATE BREAK TIME CONSTRAINTS
        print("Validating break time constraints...")
        for day in range(best_sol.tourist.days):
            if day not in best_sol.tourist.locations or len(best_sol.tourist.locations[day]) <= 1:
                continue  # Skip days with 0 or 1 attraction (no break time needed)

            # Calculate total break time for this day
            time_slots = []
            for attr in best_sol.tourist.locations[day]:
                if attr.attraction_name in best_sol.tourist.start_times[day]:
                    start_time = best_sol.tourist.start_times[day][attr.attraction_name]
                    end_time = start_time + attr.task_time
                    time_slots.append((attr, start_time, end_time))

            # Sort by start time
            time_slots.sort(key=lambda x: x[1])

            # Calculate break time
            total_break_time = 0
            for i in range(len(time_slots) - 1):
                _, _, curr_end = time_slots[i]
                _, next_start, _ = time_slots[i + 1]
                break_time = next_start - curr_end
                total_break_time += break_time

            # Fix break time if constraint violated
            if total_break_time < 1 and len(time_slots) > 1:
                constraints_valid = False
                print(f"  Break time constraint violated on day {day + 1}: {total_break_time:.1f} hours (need 1 hour)")

                # Strategy 1: Try to adjust start times to create breaks
                adjusted = False
                for i in range(len(time_slots) - 1, 0, -1):  # Work backward
                    attr, start, end = time_slots[i]
                    # Move attraction later to create a break if possible
                    needed_shift = 1.0 - total_break_time
                    # Ensure we don't exceed touring hours or attraction opening hours
                    day_end = best_sol.tourist.touring_dict[day][1]
                    attr_close = float('inf')
                    if day in attr.opening_hours:
                        attr_close = attr.opening_hours[day][1]

                    # Calculate how much we can shift this attraction
                    max_shift = min(day_end - end, attr_close - end)

                    if max_shift >= needed_shift:
                        # Adjust the start time
                        new_start = start + needed_shift
                        best_sol.tourist.start_times[day][attr.attraction_name] = new_start
                        print(f"    Adjusted start time of {attr.attraction_name} to {new_start:.1f}")
                        adjusted = True
                        break

                # Strategy 2: If adjusting times doesn't work, remove one attraction
                if not adjusted:
                    # Find lowest fun score attraction to remove
                    lowest_score = float('inf')
                    lowest_attr = None

                    for attr, _, _ in time_slots:
                        score = best_sol.get_attraction_fun_score(attr)
                        if score < lowest_score:
                            lowest_score = score
                            lowest_attr = attr

                    if lowest_attr:
                        best_sol.tourist.remove(lowest_attr)
                        best_sol.unassigned.append(lowest_attr)
                        print(f"    Removed {lowest_attr.attraction_name} to satisfy break time constraint")

        # 2. VALIDATE TRAVEL TIME CONSTRAINTS
        print("Validating travel time constraints...")
        for day in range(best_sol.tourist.days):
            if day not in best_sol.tourist.locations or not best_sol.tourist.locations[day]:
                continue

            hotel_lat = best_sol.tourist.hotel_lat
            hotel_long = best_sol.tourist.hotel_long

            # Get sorted attractions for the day
            time_slots = []
            for attr in best_sol.tourist.locations[day]:
                if attr.attraction_name in best_sol.tourist.start_times[day]:
                    start_time = best_sol.tourist.start_times[day][attr.attraction_name]
                    end_time = start_time + attr.task_time
                    time_slots.append((attr, start_time, end_time))

            time_slots.sort(key=lambda x: x[1])

            # Check travel from hotel to first attraction
            if time_slots:
                first_attr, first_start, _ = time_slots[0]
                distance = calculate_distance(
                    hotel_lat, hotel_long,
                    first_attr.lat_long[1], first_attr.lat_long[0]
                )
                travel_time = distance / 30.0  # 30 km/h average speed

                if first_start < best_sol.tourist.touring_dict[day][0] + travel_time:
                    constraints_valid = False
                    print(
                        f"  Travel time constraint violated on day {day + 1}: Cannot reach {first_attr.attraction_name} in time")

                    # Try to adjust start time
                    new_start = best_sol.tourist.touring_dict[day][0] + travel_time
                    if new_start + first_attr.task_time <= best_sol.tourist.touring_dict[day][1]:
                        best_sol.tourist.start_times[day][first_attr.attraction_name] = new_start
                        print(f"    Adjusted start time of {first_attr.attraction_name} to {new_start:.1f}")
                    else:
                        # Remove attraction if can't adjust
                        best_sol.tourist.remove(first_attr)
                        best_sol.unassigned.append(first_attr)
                        print(f"    Removed {first_attr.attraction_name} due to insufficient travel time")

            # Check travel between attractions
            for i in range(len(time_slots) - 1):
                curr_attr, _, curr_end = time_slots[i]
                next_attr, next_start, _ = time_slots[i + 1]

                distance = calculate_distance(
                    curr_attr.lat_long[1], curr_attr.lat_long[0],
                    next_attr.lat_long[1], next_attr.lat_long[0]
                )
                travel_time = distance / 30.0

                if next_start < curr_end + travel_time:
                    constraints_valid = False
                    print(
                        f"  Travel time constraint violated on day {day + 1}: Cannot travel from {curr_attr.attraction_name} to {next_attr.attraction_name} in time")

                    # Try to adjust start time of next attraction
                    new_start = curr_end + travel_time
                    if new_start + next_attr.task_time <= best_sol.tourist.touring_dict[day][1]:
                        best_sol.tourist.start_times[day][next_attr.attraction_name] = new_start
                        print(f"    Adjusted start time of {next_attr.attraction_name} to {new_start:.1f}")
                    else:
                        # Remove next attraction if can't adjust
                        best_sol.tourist.remove(next_attr)
                        best_sol.unassigned.append(next_attr)
                        print(f"    Removed {next_attr.attraction_name} due to insufficient travel time")

            # Check travel from last attraction back to hotel
            if time_slots:
                last_attr, _, last_end = time_slots[-1]
                distance = calculate_distance(
                    last_attr.lat_long[1], last_attr.lat_long[0],
                    hotel_lat, hotel_long
                )
                travel_time = distance / 30.0

                if last_end + travel_time > best_sol.tourist.touring_dict[day][1]:
                    constraints_valid = False
                    print(
                        f"  Travel time constraint violated on day {day + 1}: Cannot return to hotel from {last_attr.attraction_name} in time")
                    best_sol.tourist.remove(last_attr)
                    best_sol.unassigned.append(last_attr)
                    print(f"    Removed {last_attr.attraction_name} due to insufficient travel time back to hotel")

        # 3. VALIDATE OPENING HOURS CONSTRAINTS
        print("Validating opening hours constraints...")
        for day in range(best_sol.tourist.days):
            if day not in best_sol.tourist.locations:
                continue

            for attr in list(best_sol.tourist.locations[day]):  # Use list to allow removal during iteration
                if attr.attraction_name not in best_sol.tourist.start_times[day]:
                    continue

                start_time = best_sol.tourist.start_times[day][attr.attraction_name]
                end_time = start_time + attr.task_time

                # Check if day is in opening hours
                if day not in attr.opening_hours:
                    constraints_valid = False
                    print(f"  Opening hours constraint violated on day {day + 1}: {attr.attraction_name} is closed")
                    best_sol.tourist.remove(attr)
                    best_sol.unassigned.append(attr)
                    print(f"    Removed {attr.attraction_name} due to attraction being closed on this day")
                    continue

                opening, closing = attr.opening_hours[day]

                # Check if time is within opening hours
                if start_time < opening or end_time > closing:
                    constraints_valid = False
                    print(
                        f"  Opening hours constraint violated on day {day + 1}: {attr.attraction_name} visit outside opening hours")

                    # Try to adjust start time
                    if opening + attr.task_time <= closing:
                        best_sol.tourist.start_times[day][attr.attraction_name] = opening
                        print(f"    Adjusted start time of {attr.attraction_name} to {opening:.1f}")
                    else:
                        # Remove if can't adjust
                        best_sol.tourist.remove(attr)
                        best_sol.unassigned.append(attr)
                        print(f"    Removed {attr.attraction_name} due to opening hours constraint")

                # Check if outside touring hours
                touring_start, touring_end = best_sol.tourist.touring_dict[day]
                if start_time < touring_start or end_time > touring_end:
                    constraints_valid = False
                    print(
                        f"  Touring hours constraint violated on day {day + 1}: {attr.attraction_name} outside touring hours")

                    # Try to adjust start time
                    if touring_start + attr.task_time <= touring_end:
                        best_sol.tourist.start_times[day][attr.attraction_name] = touring_start
                        print(f"    Adjusted start time of {attr.attraction_name} to {touring_start:.1f}")
                    else:
                        # Remove if can't adjust
                        best_sol.tourist.remove(attr)
                        best_sol.unassigned.append(attr)
                        print(f"    Removed {attr.attraction_name} due to touring hours constraint")

        # 4. VALIDATE OVERLAPPING VISITS CONSTRAINT
        print("Validating overlapping visits constraint...")
        for day in range(best_sol.tourist.days):
            if day not in best_sol.tourist.locations or len(best_sol.tourist.locations[day]) <= 1:
                continue

            # Get time slots
            time_slots = []
            for attr in best_sol.tourist.locations[day]:
                if attr.attraction_name in best_sol.tourist.start_times[day]:
                    start_time = best_sol.tourist.start_times[day][attr.attraction_name]
                    end_time = start_time + attr.task_time
                    time_slots.append((attr, start_time, end_time))

            # Sort by start time
            time_slots.sort(key=lambda x: x[1])

            # Check for overlaps
            for i in range(len(time_slots) - 1):
                curr_attr, curr_start, curr_end = time_slots[i]
                next_attr, next_start, next_end = time_slots[i + 1]

                if curr_end > next_start:
                    constraints_valid = False
                    print(
                        f"  Overlapping visits constraint violated on day {day + 1}: {curr_attr.attraction_name} and {next_attr.attraction_name}")

                    # Try to adjust start time of next attraction
                    new_start = curr_end
                    if new_start + next_attr.task_time <= best_sol.tourist.touring_dict[day][1]:
                        best_sol.tourist.start_times[day][next_attr.attraction_name] = new_start
                        print(f"    Adjusted start time of {next_attr.attraction_name} to {new_start:.1f}")
                    else:
                        # Remove next attraction if can't adjust
                        best_sol.tourist.remove(next_attr)
                        best_sol.unassigned.append(next_attr)
                        print(f"    Removed {next_attr.attraction_name} to resolve overlap")

        # 5. VALIDATE CAPACITY CONSTRAINT (MAX 3 ATTRACTIONS PER DAY)
        print("Validating capacity constraint...")
        for day in range(best_sol.tourist.days):
            if day not in best_sol.tourist.locations:
                continue

            if len(best_sol.tourist.locations[day]) > 3:
                constraints_valid = False
                print(
                    f"  Capacity constraint violated on day {day + 1}: {len(best_sol.tourist.locations[day])} attractions (max 3)")

                # Sort attractions by fun score (descending)
                attractions_by_score = [(attr, best_sol.get_attraction_fun_score(attr))
                                        for attr in best_sol.tourist.locations[day]]
                attractions_by_score.sort(key=lambda x: x[1], reverse=True)

                # Keep top 3, remove others
                for attr, _ in attractions_by_score[3:]:
                    best_sol.tourist.remove(attr)
                    best_sol.unassigned.append(attr)
                    print(f"    Removed {attr.attraction_name} to satisfy capacity constraint")

        # 6. VALIDATE BUDGET CONSTRAINT
        if best_sol.tourist.money_spent > best_sol.tourist.budget:
            constraints_valid = False
            print(
                f"  Budget constraint violated: Spent ${best_sol.tourist.money_spent} but budget is ${best_sol.tourist.budget}")

            # Calculate how much we need to save
            excess = best_sol.tourist.money_spent - best_sol.tourist.budget

            # Get all attractions with costs and fun scores
            all_attractions = []
            for day in range(best_sol.tourist.days):
                if day in best_sol.tourist.locations:
                    for attr in best_sol.tourist.locations[day]:
                        fun_score = best_sol.get_attraction_fun_score(attr)
                        efficiency = fun_score / attr.cost if attr.cost > 0 else float('inf')
                        all_attractions.append((attr, attr.cost, efficiency))

            # Sort by efficiency (ascending - remove least efficient first)
            all_attractions.sort(key=lambda x: x[2])

            # Remove attractions until budget constraint satisfied
            saved = 0
            for attr, cost, _ in all_attractions:
                best_sol.tourist.remove(attr)
                best_sol.unassigned.append(attr)
                saved += cost
                print(f"    Removed {attr.attraction_name} (cost: ${cost}) to satisfy budget constraint")

                if saved >= excess:
                    break

        # Update state after fixes
        best_sol.update_included_categories()

        # If all constraints are valid, we can break out of the loop
        if constraints_valid:
            print("All constraints validated successfully!")
            break

        # If we've made it to the last iteration and still have violations
        if iteration == max_iterations - 1 and not constraints_valid:
            print("WARNING: Could not fix all constraint violations within maximum iterations.")

    # Recalculate final objective after fixing constraints using the evaluation module
    best_sol.update_included_categories()
    optimized_metrics = calculate_objective_metrics(best_sol)

    # Get standardized comparison
    comparison = compare_solutions(
        initial_metrics, 
        optimized_metrics,
        "Initial", 
        "Optimized"
    )

    print(f"\nFinal solution objective after constraint validation (negative): {optimized_metrics['objective_negative']:.4f}")
    print(f"Final solution objective after constraint validation (positive): {optimized_metrics['objective_positive']:.4f}")
    print(f"Improvement: {comparison['improvement_description']}")

    # Create summary comparing initial and optimized solutions
    create_tourist_summary(tourist.idx, initial_solution, best_sol)

    # Save optimized solution
    save_smjsp_output(f"Tourist_{tourist.idx}_Final_Optimized_Itinerary", best_sol, "solution")

    # Save serialized optimized solution to stage1_results
    results_dir = ensure_results_directory()
    optimized_json_path = os.path.join(results_dir, f"Tourist_{tourist.idx}_optimized_solution.json")
    best_sol.save_to_file(optimized_json_path)

    return {"initial": initial_solution, "optimised": best_sol}

def create_tourist_summary(tourist_id, initial_solution, optimized_solution, 
                        initial_metrics=None, optimized_metrics=None, comparison=None):
    """Create a comprehensive summary for an individual tourist."""
    
    # If metrics aren't provided, calculate them
    if initial_metrics is None:
        initial_metrics = calculate_objective_metrics(initial_solution)
    
    if optimized_metrics is None:
        optimized_metrics = calculate_objective_metrics(optimized_solution)
    
    if comparison is None:
        comparison = compare_solutions(initial_metrics, optimized_metrics)
    
    # Create output file in stage1_results
    results_dir = ensure_results_directory()
    filename = os.path.join(results_dir, f"Tourist_{tourist_id}_Summary.txt")

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
        f.write(f"Initial Solution: {initial_metrics['objective_positive']:.4f}\n")
        f.write(f"Optimized Solution: {optimized_metrics['objective_positive']:.4f}\n")
        f.write(f"Improvement: {comparison['improvement_description']}\n\n")
        
        # Component Breakdown
        f.write("NORMALIZED COMPONENT VALUES\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Component':<20} {'Initial':<15} {'Optimized':<15} {'Change (%)':<10}\n")
        
        # Use metrics from evaluation module
        for component, label in [
            ("normalized_fun", "Fun Score"),
            ("normalized_distance", "Distance"),
            ("normalized_attractions", "Attractions")
        ]:
            change = comparison["component_changes"][component]
            f.write(f"{label:<20} {initial_metrics[component]:<15.4f} {optimized_metrics[component]:<15.4f} {change:+.1f}%\n")
        
        f.write("\n")
        
        # Raw metrics
        f.write("RAW METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Metric':<20} {'Initial':<15} {'Optimized':<15}\n")
        f.write(f"{'Attractions Count':<20} {initial_metrics['attractions_count']:<15d} {optimized_metrics['attractions_count']:<15d}\n")
        f.write(f"{'Total Distance (km)':<20} {initial_metrics['distance']:<15.2f} {optimized_metrics['distance']:<15.2f}\n")
        f.write(f"{'Raw Fun Score':<20} {initial_metrics['fun_score']:<15.2f} {optimized_metrics['fun_score']:<15.2f}\n")
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




def create_aggregate_summary(results, tourist_count):
    """
    Creates an aggregate summary for multiple tourists.
    Used only in aggregate mode to summarize performance across all tourists.
    """
    results_dir = ensure_results_directory()
    filename = os.path.join(results_dir, "aggregate_tourist_summary.txt")

    # Calculate aggregate statistics
    initial_objectives = []
    optimized_objectives = []
    preference_groups = {}

    for tourist_id, data in results.items():
        initial_bd = get_objective_breakdown(data["initial"])
        optimized_bd = get_objective_breakdown(data["optimized"])

        initial_obj = initial_bd['weighted_objective_positive']
        optimized_obj = optimized_bd['weighted_objective_positive']

        initial_objectives.append(initial_obj)
        optimized_objectives.append(optimized_obj)

        # Group by preference
        pref_key = '/'.join(sorted(data["initial"].tourist.preferences))
        if pref_key not in preference_groups:
            preference_groups[pref_key] = {"initial": [], "optimized": []}
        preference_groups[pref_key]["initial"].append(initial_obj)
        preference_groups[pref_key]["optimized"].append(optimized_obj)

    # Calculate averages
    avg_initial = sum(initial_objectives) / len(initial_objectives) if initial_objectives else 0
    avg_optimized = sum(optimized_objectives) / len(optimized_objectives) if optimized_objectives else 0

    # Calculate improvement
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

    # Write summary
    with open(filename, 'w') as f:
        f.write("AGGREGATE TOURIST ITINERARY PLANNING SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total Tourists: {tourist_count}\n\n")

        f.write("OVERALL PERFORMANCE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Average Initial Objective: {avg_initial:.4f}\n")
        f.write(f"Average Optimized Objective: {avg_optimized:.4f}\n")
        f.write(f"Overall Improvement: {improvement_desc}\n\n")

        # Performance by preference group
        f.write("PERFORMANCE BY TOURIST PREFERENCE\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Preference':<25} {'Initial':<15} {'Optimized':<15} {'Improvement':<15}\n")

        for pref, data in preference_groups.items():
            pref_initial = sum(data["initial"]) / len(data["initial"]) if data["initial"] else 0
            pref_optimized = sum(data["optimized"]) / len(data["optimized"]) if data["optimized"] else 0

            # Calculate improvement for this preference group
            if pref_initial < 0 and pref_optimized < 0:
                pref_improvement = f"{(abs(pref_initial) - abs(pref_optimized)) / abs(pref_initial) * 100:.1f}% reduced"
            elif pref_initial < 0 and pref_optimized >= 0:
                pref_improvement = "Complete reversal"
            else:
                pref_improvement = f"{(pref_optimized - pref_initial) / abs(pref_initial) * 100:.1f}%"

            f.write(f"{pref:<25} {pref_initial:<15.4f} {pref_optimized:<15.4f} {pref_improvement:<15}\n")

        f.write(f"\nSummary generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"Created aggregate summary: {filename}")
    return filename

def create_enhanced_summary(method_objectives, method_components, preference_groups, count, dr_alns = False):
    """Create enhanced aggregate summary for Stage 1 results."""
    filename = "enhanced_tourist_comparison_summary.txt"
    if dr_alns:
        filename = "stage2_results/enhanced_tourist_comparison_summary_DR_ALNS.txt"
    
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
            new_opening[dayk - 1] = hourset
        a_.opening_hours = new_opening

    # Choose tourist
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

    print("\nItinerary planning complete!")


def run_aggregate_results(attraction_csv, tourist_csv, seed):
    """
    Perform aggregated analysis across all tourists and save results to stage1_results directory.
    """
    # Parse data first
    parser = Parser(attraction_csv, tourist_csv)

    # Adjust attraction opening hours keys (assume original keys are 1-indexed)
    for attraction in parser.attractions:
        new_opening = {}
        for day, hours in attraction.opening_hours.items():
            new_opening[day - 1] = hours
        attraction.opening_hours = new_opening

    all_results = {}
    tourist_count = 0

    # Process each tourist
    for tourist in parser.tourists:
        print(f"Processing Tourist ID: {tourist.idx} ({tourist_count + 1}/{len(parser.tourists)})")

        # Generate and optimize solution
        solution = solve_single_tourist(tourist, parser.attractions, seed)
        all_results[tourist.idx] = solution
        tourist_count += 1

    # Create aggregate summary
    create_aggregate_summary(all_results, tourist_count)

    print(f"Completed aggregate analysis for {tourist_count} tourists.")


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
    parser.add_argument("--diversity_bonus", type=float, default=15,
                        help="bonus points for first attraction of each category")
    parser.add_argument("--fun_weight", type=float, default=0.6,
                        help="weight for fun score in objective function")
    parser.add_argument("--distance_weight", type=float, default=0.3,
                        help="weight for distance in objective function")
    parser.add_argument("--attraction_weight", type=float, default=0.5,
                        help="weight for number of attractions in objective function")
    parser.add_argument("--save_initial", action="store_true",
                        help="Save initial solutions (default: only save optimized)")

    args = parser.parse_args()

    # Import json for JSON serialization
    import json
    import random

    # Weights
    w_list = [args.fun_weight, args.distance_weight, args.attraction_weight]
    s_ = sum(w_list)
    if s_ > 0:
        w_norm = [v / s_ for v in w_list]
    else:
        w_norm = w_list

    # Create the stage1_results directory before running any operations
    ensure_results_directory()

    if args.mode == "single":
        run_single_tourist(args.attraction_data, args.tourist_data, args.seed, args.tourist_id)
    elif args.mode == "aggregate":
        # Perform aggregated analysis across all tourists
        run_aggregate_results(args.attraction_data, args.tourist_data, args.seed)
    else:
        print("Invalid mode. Please use 'aggregate' or 'single'.")

