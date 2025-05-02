"""
Stage 2: Disruption Handling and Comparison
Compares how different planning strategies respond to identical disruptions.
"""

import argparse
import json
import os
import sys
import copy
import random
import math
import numpy.random as rnd

from disruptions import (
    rainy_day_attraction, heat_wave_attraction, early_closure,
    nothing_happens_attraction, family_day
)
from operators import *
from rcjsp import Parser, SMJSP
from src.alns import ALNS
from src.alns.criteria import SimulatedAnnealing
from utils import calculate_distance
from evaluation import calculate_objective_metrics, compare_solutions
from rcjsp import create_enhanced_smjsp, EnhancedObjectiveSMJSP


def save_solution_to_json(solution, filename, scenario_id, forecasts):
    """
    Save SMJSP solution to a JSON file with detailed itinerary information.

    Args:
        solution: SMJSP solution object
        filename: Output filename (including path)
        scenario_id: ID of the scenario
        forecasts: Dictionary of forecasts for each day
    """
    # Clean preferences (remove single quotes and whitespace)
    clean_preferences = []
    for pref in solution.tourist.preferences:
        # Make sure to properly handle preferences that might have single quotes
        if isinstance(pref, str):
            # First strip whitespace, then remove any remaining single quotes
            clean_pref = pref.strip().replace("'", "")
            clean_preferences.append(clean_pref)
        else:
            clean_preferences.append(pref)

    solution_data = {
        'scenario_info': {
            'scenario_id': scenario_id,
            'forecasts': forecasts
        },
        'tourist': {
            'idx': solution.tourist.idx,
            'preferences': clean_preferences,  # Fixed preferences format
            'budget': solution.tourist.budget,
            'days': solution.tourist.days,
            'touring_hours': solution.tourist.touring_hours,
            'hotel_long': solution.tourist.hotel_long,
            'hotel_lat': solution.tourist.hotel_lat,
            'money_spent': solution.tourist.money_spent,
            'locations': {},
            'start_times': {}
        },
        'included_categories': list(solution.included_categories),
        'unassigned': []
    }

    # Save locations and start times
    for day in solution.tourist.locations:
        # Convert attraction objects to attraction names (for serialization)
        solution_data['tourist']['locations'][day] = [attr.attraction_name for attr in solution.tourist.locations[day]]
        solution_data['tourist']['start_times'][day] = solution.tourist.start_times[day]

    # Save unassigned attractions (by name)
    solution_data['unassigned'] = [attr.attraction_name for attr in solution.unassigned]

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save to file
    with open(filename, 'w') as f:
        json.dump(solution_data, f, indent=2)


def generate_disruption_scenarios(solution, num_scenarios=3, seed=None):
    """
    Generate predefined disruption scenarios, ensuring disruptions
    affect 30% of days that have attractions planned.

    Args:
        solution: SMJSP solution object with planned itinerary
        num_scenarios: Number of scenarios to generate
        seed: Random seed for reproducibility

    Returns:
        List of scenario dictionaries
    """
    if seed:
        random.seed(seed)

    days_count = solution.tourist.days
    scenarios = []

    for scenario_idx in range(num_scenarios):
        disruptions = []
        forecasts = {}

        # Identify days that have attractions planned
        planned_days = [d for d in range(days_count)
                        if d in solution.tourist.locations and solution.tourist.locations[d]]

        if not planned_days:
            # If no days have attractions, just pick random days
            min_disruption_days = max(1, math.ceil(days_count * 0.3))
            disruption_days = random.sample(range(days_count), min(min_disruption_days, days_count))
        else:
            # Target 30% of planned days for disruptions
            min_disruption_days = max(1, math.ceil(len(planned_days) * 0.3))
            # Select from days that actually have attractions
            disruption_days = random.sample(planned_days, min(min_disruption_days, len(planned_days)))

        # Define forecasts for all days
        for d in range(days_count):
            if d in disruption_days:
                # Force a disruption (anything except normal day)
                forecast_type = random.choice(["rainy", "heat_wave", "early_closure", "family_day"])
            else:
                weights = [0.6, 0.1, 0.1, 0.1, 0.1]  # Bias toward normal days
                forecast_type = random.choices(
                    ["normal", "rainy", "heat_wave", "early_closure", "family_day"],
                    weights=weights, k=1)[0]

            forecasts[d] = forecast_type
            disruptions.append((d, forecast_type))

        scenarios.append({
            "scenario": scenario_idx,
            "disruptions": disruptions,
            "forecasts": forecasts,
            "disrupted_days": disruption_days
        })

    return scenarios


def apply_disruptions(solution, disruptions):
    """
    Apply a predefined set of disruptions to a solution with enhanced constraint handling
    for all disruption types, using functions from disruptions.py.

    Args:
        solution: SMJSP solution object
        disruptions: List of (day, disruption_type) tuples

    Returns:
        SMJSP solution with disruptions applied
    """
    # Map disruption names to functions
    disruption_map = {
        "normal": nothing_happens_attraction,
        "rainy": rainy_day_attraction,
        "heat_wave": heat_wave_attraction,
        "early_closure": early_closure,
        "family_day": family_day
    }

    # Clone the solution to avoid modifying the original
    disrupted_solution = copy.deepcopy(solution)

    # Apply each disruption
    for day_str, disruption_type in disruptions:
        day = int(day_str)
        disruption_func = disruption_map.get(disruption_type.lower(), nothing_happens_attraction)

        # Apply the disruption function to modify attraction opening hours
        disrupted_solution.attractions = disruption_func(disrupted_solution.attractions, day)

        # Process attractions in the tourist's schedule based on the disruption type
        if day in disrupted_solution.tourist.locations:
            to_remove = []

            for attr in disrupted_solution.tourist.locations[day]:
                # Find matching attraction in updated list
                matching = [a for a in disrupted_solution.attractions if a.attraction_name == attr.attraction_name]

                # If attraction is no longer available, remove it
                if not matching or day not in matching[0].opening_hours:
                    to_remove.append(attr)
                    continue

                forecast_type = disruption_type.lower()

                # Handle specific disruption constraints
                if forecast_type == "rainy" and "Outdoor" in attr.categories:
                    to_remove.append(attr)

                elif forecast_type == "heat_wave" and any(
                        cat in attr.categories for cat in ["Sporty", "Nature", "Outdoor"]):
                    to_remove.append(attr)

                elif forecast_type == "early_closure" and "Outdoor" not in attr.categories:
                    start_time = disrupted_solution.tourist.start_times[day].get(attr.attraction_name)
                    if start_time is not None:
                        # Check if tour starts at or after noon
                        touring_start = disrupted_solution.tourist.touring_dict[day][0]

                        if touring_start >= 12:
                            to_remove.append(attr)
                        else:
                            # Check if visit extends past noon
                            end_time = start_time + attr.task_time
                            if end_time > 12:
                                to_remove.append(attr)

                elif forecast_type == "family_day" and "Family" not in attr.categories:
                    to_remove.append(attr)

            # Remove all affected attractions
            for attr in to_remove:
                disrupted_solution.tourist.remove(attr)
                disrupted_solution.unassigned.append(attr)

    return disrupted_solution


def quick_repair(solution, tourist_id, num_scenarios=3, seed=None):
    """
    This solution performs dynamic replanning (quick repair)
    under disruptions. It applies simpler repair operators to fix
    the disrupted solution without using the full ALNS framework.

    Args:
        solution: Original SMJSP solution object
        tourist_id: Tourist ID for saving files
        num_scenarios: Number of disruption scenarios to simulate
        seed: Random seed for reproducibility

    Returns:
        A list of scenario result dictionaries and repaired solutions
    """
    if seed is not None:
        random.seed(seed)

    # Calculate original objective value
    orig_obj_neg = solution.objective()
    orig_obj_pos = -orig_obj_neg
    print("\nEvaluating dynamic replanning with weather forecasts (Quick Repair)...")
    print(f"Original solution objective (positive): {orig_obj_pos:.4f}")

    # Generate disruption scenarios
    scenarios = generate_disruption_scenarios(solution, num_scenarios=num_scenarios, seed=seed)
    scenario_results = []
    repaired_solutions = {}

    os.makedirs("disruption_routes", exist_ok=True)

    for scenario in scenarios:
        print(f"\nScenario {scenario['scenario']}:")

        # Print forecast for each day
        for d, forecast in scenario["forecasts"].items():
            print(f" Day {d + 1} forecast: {forecast.capitalize()}")

        # Apply disruptions
        disrupted_solution = apply_disruptions(solution, scenario["disruptions"])

        # Create a repair seed based on scenario index
        dd_seed = seed + scenario['scenario'] if seed is not None else random.randint(0, 9999)
        rr = rnd.RandomState(dd_seed)

        # Apply repair operators
        repaired_solution = repair_balanced(disrupted_solution, rr)
        repaired_solution = repair_greedy(repaired_solution, rr)

        # Post-process to ensure all constraints are met
        print(" Performing comprehensive post-processing to ensure all constraints are met...")
        max_iterations = 3
        
        for iteration in range(max_iterations):
            # Track if any constraint violations were fixed this iteration
            fixed_violations = False
            
            # 1. VALIDATE AND FIX DISRUPTION-SPECIFIC CONSTRAINTS
            for day_str, forecast in scenario["forecasts"].items():
                day = int(day_str)
                forecast_type = forecast.lower()

                if day not in repaired_solution.tourist.locations:
                    continue

                # Process attractions for this day
                attractions_to_remove = []
                for attr in repaired_solution.tourist.locations[day]:
                    # Apply specific constraints based on forecast type
                    if forecast_type == "rainy" and "Outdoor" in attr.categories:
                        attractions_to_remove.append(attr)
                        print(f"   Removing {attr.attraction_name} on day {day + 1}: Outdoor attraction during rainy day")
                        fixed_violations = True

                    elif forecast_type == "heat_wave" and any(
                            cat in attr.categories for cat in ["Sporty", "Nature", "Outdoor"]):
                        attractions_to_remove.append(attr)
                        print(f"   Removing {attr.attraction_name} on day {day + 1}: {', '.join(set(attr.categories) & set(['Sporty', 'Nature', 'Outdoor']))} attraction during heat wave")
                        fixed_violations = True

                    elif forecast_type == "early_closure" and "Outdoor" not in attr.categories:
                        start_time = repaired_solution.tourist.start_times[day].get(attr.attraction_name)
                        if start_time is not None:
                            # For early closure, indoor attractions close at noon
                            if start_time >= 12:
                                attractions_to_remove.append(attr)
                                print(f"   Removing {attr.attraction_name} on day {day + 1}: starts at {start_time} (after noon)")
                                fixed_violations = True
                            else:
                                # If it extends past noon, remove it
                                end_time = start_time + attr.task_time
                                if end_time > 12:
                                    attractions_to_remove.append(attr)
                                    print(f"   Removing {attr.attraction_name} on day {day + 1}: ends at {end_time} (after noon)")
                                    fixed_violations = True

                    elif forecast_type == "family_day" and "Family" not in attr.categories:
                        attractions_to_remove.append(attr)
                        print(f"   Removing {attr.attraction_name} on day {day + 1}: not family-friendly during family day")
                        fixed_violations = True

                # Remove all problematic attractions
                for attr in attractions_to_remove:
                    repaired_solution.tourist.remove(attr)
                    repaired_solution.unassigned.append(attr)

            # 2. VALIDATE AND FIX BREAK TIME CONSTRAINTS
            for day in range(repaired_solution.tourist.days):
                if day not in repaired_solution.tourist.locations or len(repaired_solution.tourist.locations[day]) <= 1:
                    continue

                # Calculate time slots
                time_slots = []
                for attr in repaired_solution.tourist.locations[day]:
                    if attr.attraction_name in repaired_solution.tourist.start_times[day]:
                        start_time = repaired_solution.tourist.start_times[day][attr.attraction_name]
                        end_time = start_time + attr.task_time
                        time_slots.append((attr, start_time, end_time))

                # Sort by start time
                time_slots.sort(key=lambda x: x[1])

                # Calculate total break time
                total_break_time = 0
                for i in range(len(time_slots) - 1):
                    _, _, curr_end = time_slots[i]
                    _, next_start, _ = time_slots[i + 1]
                    break_time = next_start - curr_end
                    total_break_time += break_time

                # If break time constraint is violated
                if total_break_time < 1 and len(time_slots) > 1:
                    print(f"   Break time constraint violated on day {day + 1}: {total_break_time:.1f} hours (need 1 hour)")
                    
                    # Strategy 1: Try to adjust start times to create breaks
                    adjusted = False
                    for i in range(len(time_slots) - 1, 0, -1):  # Work backward
                        attr, start, end = time_slots[i]
                        # Move attraction later to create a break if possible
                        needed_shift = 1.0 - total_break_time
                        # Ensure we don't exceed touring hours or attraction opening hours
                        day_end = repaired_solution.tourist.touring_dict[day][1]
                        attr_close = float('inf')
                        if day in attr.opening_hours:
                            attr_close = attr.opening_hours[day][1]

                        # Calculate how much we can shift this attraction
                        max_shift = min(day_end - end, attr_close - end)

                        if max_shift >= needed_shift:
                            # Adjust the start time
                            new_start = start + needed_shift
                            repaired_solution.tourist.start_times[day][attr.attraction_name] = new_start
                            print(f"    Adjusted start time of {attr.attraction_name} to {new_start:.1f}")
                            adjusted = True
                            fixed_violations = True
                            break
                    
                    # Strategy 2: If adjusting times doesn't work, remove lowest fun score attraction
                    if not adjusted:
                        # Sort by fun score (ascending)
                        scored_attractions = [(attr, repaired_solution.get_attraction_fun_score(attr))
                                            for attr, _, _ in time_slots]
                        scored_attractions.sort(key=lambda x: x[1])
                        
                        if scored_attractions:
                            attr_to_remove = scored_attractions[0][0]
                            repaired_solution.tourist.remove(attr_to_remove)
                            repaired_solution.unassigned.append(attr_to_remove)
                            print(f"    Removed {attr_to_remove.attraction_name} to satisfy break time constraint")
                            fixed_violations = True

            # 3. VALIDATE AND FIX TRAVEL TIME CONSTRAINTS
            for day in range(repaired_solution.tourist.days):
                if day not in repaired_solution.tourist.locations or not repaired_solution.tourist.locations[day]:
                    continue

                hotel_lat = repaired_solution.tourist.hotel_lat
                hotel_long = repaired_solution.tourist.hotel_long
                
                # Convert to list with start/end times
                activities = []
                for attr in repaired_solution.tourist.locations[day]:
                    if attr.attraction_name in repaired_solution.tourist.start_times[day]:
                        start_time = repaired_solution.tourist.start_times[day][attr.attraction_name]
                        end_time = start_time + attr.task_time
                        activities.append((attr, start_time, end_time))
                
                activities.sort(key=lambda x: x[1])  # Sort by start time
                
                if not activities:
                    continue
                
                # Check travel time from hotel to first attraction
                first_attr, first_start, _ = activities[0]
                distance = calculate_distance(
                    hotel_lat, hotel_long,
                    first_attr.lat_long[1], first_attr.lat_long[0]
                )
                travel_time = distance / 30.0  # 30 km/h average speed
                
                if first_start < repaired_solution.tourist.touring_dict[day][0] + travel_time:
                    print(f"   Travel time constraint violated on day {day + 1}: Cannot reach first attraction in time")
                    # Try to adjust start time
                    new_start = repaired_solution.tourist.touring_dict[day][0] + travel_time
                    if new_start + first_attr.task_time <= repaired_solution.tourist.touring_dict[day][1]:
                        repaired_solution.tourist.start_times[day][first_attr.attraction_name] = new_start
                        print(f"    Adjusted start time of {first_attr.attraction_name} to {new_start:.1f}")
                        fixed_violations = True
                    else:
                        # Remove attraction if can't adjust
                        repaired_solution.tourist.remove(first_attr)
                        repaired_solution.unassigned.append(first_attr)
                        print(f"    Removed {first_attr.attraction_name} due to insufficient travel time")
                        fixed_violations = True
                        continue  # Skip to next day since we modified activities
                
                # Check travel time between attractions
                for i in range(len(activities) - 1):
                    curr_attr, _, curr_end = activities[i]
                    next_attr, next_start, _ = activities[i + 1]
                    
                    distance = calculate_distance(
                        curr_attr.lat_long[1], curr_attr.lat_long[0],
                        next_attr.lat_long[1], next_attr.lat_long[0]
                    )
                    travel_time = distance / 30.0
                    
                    if next_start < curr_end + travel_time:
                        print(f"   Travel time constraint violated on day {day + 1}: Cannot travel from {curr_attr.attraction_name} to {next_attr.attraction_name} in time")
                        # Try to adjust next attraction's start time
                        new_start = curr_end + travel_time
                        if new_start + next_attr.task_time <= repaired_solution.tourist.touring_dict[day][1]:
                            repaired_solution.tourist.start_times[day][next_attr.attraction_name] = new_start
                            print(f"    Adjusted start time of {next_attr.attraction_name} to {new_start:.1f}")
                            fixed_violations = True
                        else:
                            # Remove next attraction if can't adjust
                            repaired_solution.tourist.remove(next_attr)
                            repaired_solution.unassigned.append(next_attr)
                            print(f"    Removed {next_attr.attraction_name} due to insufficient travel time")
                            fixed_violations = True
                            break  # Exit this loop since we modified activities
                
                # Reload activities since we might have modified the list
                activities = []
                for attr in repaired_solution.tourist.locations[day]:
                    if attr.attraction_name in repaired_solution.tourist.start_times[day]:
                        start_time = repaired_solution.tourist.start_times[day][attr.attraction_name]
                        end_time = start_time + attr.task_time
                        activities.append((attr, start_time, end_time))
                
                activities.sort(key=lambda x: x[1])
                
                if not activities:
                    continue
                
                # Check travel time from last attraction back to hotel
                last_attr, _, last_end = activities[-1]
                distance = calculate_distance(
                    last_attr.lat_long[1], last_attr.lat_long[0],
                    hotel_lat, hotel_long
                )
                travel_time = distance / 30.0
                
                if last_end + travel_time > repaired_solution.tourist.touring_dict[day][1]:
                    print(f"   Travel time constraint violated on day {day + 1}: Cannot return to hotel in time")
                    # Since this is the last activity, we just remove it
                    repaired_solution.tourist.remove(last_attr)
                    repaired_solution.unassigned.append(last_attr)
                    print(f"    Removed {last_attr.attraction_name} due to insufficient travel time back to hotel")
                    fixed_violations = True
            
            # 4. VALIDATE AND FIX CAPACITY CONSTRAINTS (MAX 3 ATTRACTIONS PER DAY)
            for day in range(repaired_solution.tourist.days):
                if day not in repaired_solution.tourist.locations:
                    continue
                
                if len(repaired_solution.tourist.locations[day]) > 3:
                    print(f"   Capacity constraint violated on day {day + 1}: {len(repaired_solution.tourist.locations[day])} attractions (max is 3)")
                    
                    # Score attractions by fun value
                    scored_attractions = [(attr, repaired_solution.get_attraction_fun_score(attr))
                                        for attr in repaired_solution.tourist.locations[day]]
                    scored_attractions.sort(key=lambda x: x[1], reverse=True)  # Sort by descending fun score
                    
                    # Keep top 3, remove the rest
                    for attr, _ in scored_attractions[3:]:
                        repaired_solution.tourist.remove(attr)
                        repaired_solution.unassigned.append(attr)
                        print(f"    Removed {attr.attraction_name} to satisfy capacity constraint")
                        fixed_violations = True
            
            # 5. VALIDATE AND FIX BUDGET CONSTRAINTS
            if repaired_solution.tourist.money_spent > repaired_solution.tourist.budget:
                print(f"   Budget constraint violated: Spent ${repaired_solution.tourist.money_spent} exceeds budget ${repaired_solution.tourist.budget}")
                
                # Find all attractions with their costs and fun scores
                all_attractions = []
                for day in range(repaired_solution.tourist.days):
                    if day in repaired_solution.tourist.locations:
                        for attr in repaired_solution.tourist.locations[day]:
                            fun_score = repaired_solution.get_attraction_fun_score(attr)
                            cost_efficiency = fun_score / (attr.cost if attr.cost > 0 else 0.1)  # Avoid division by zero
                            all_attractions.append((attr, attr.cost, cost_efficiency))
                
                # Sort by cost efficiency (ascending - remove least efficient first)
                all_attractions.sort(key=lambda x: x[2])
                
                # Remove attractions until budget is satisfied
                excess = repaired_solution.tourist.money_spent - repaired_solution.tourist.budget
                saved = 0
                for attr, cost, _ in all_attractions:
                    if saved >= excess:
                        break
                    repaired_solution.tourist.remove(attr)
                    repaired_solution.unassigned.append(attr)
                    saved += cost
                    print(f"    Removed {attr.attraction_name} (cost: ${cost}) to satisfy budget constraint")
                    fixed_violations = True
            
            # 6. VALIDATE AND FIX OPENING HOURS CONSTRAINTS
            for day in range(repaired_solution.tourist.days):
                if day not in repaired_solution.tourist.locations:
                    continue
                
                for attr in list(repaired_solution.tourist.locations[day]):  # Use list to safely modify during iteration
                    if attr.attraction_name not in repaired_solution.tourist.start_times.get(day, {}):
                        continue
                    
                    start_time = repaired_solution.tourist.start_times[day][attr.attraction_name]
                    end_time = start_time + attr.task_time
                    
                    # Check if day is in opening hours
                    if day not in attr.opening_hours:
                        print(f"   Opening hours constraint violated on day {day + 1}: {attr.attraction_name} is closed")
                        repaired_solution.tourist.remove(attr)
                        repaired_solution.unassigned.append(attr)
                        print(f"    Removed {attr.attraction_name} due to attraction being closed on this day")
                        fixed_violations = True
                        continue
                    
                    opening, closing = attr.opening_hours[day]
                    
                    # Check if visit is within opening hours
                    if start_time < opening or end_time > closing:
                        print(f"   Opening hours constraint violated on day {day + 1}: {attr.attraction_name} outside opening hours ({opening}-{closing})")
                        
                        # Try to adjust start time
                        if opening + attr.task_time <= closing:
                            new_start = opening
                            repaired_solution.tourist.start_times[day][attr.attraction_name] = new_start
                            print(f"    Adjusted start time of {attr.attraction_name} to {new_start}")
                            fixed_violations = True
                        else:
                            # Remove if can't adjust
                            repaired_solution.tourist.remove(attr)
                            repaired_solution.unassigned.append(attr)
                            print(f"    Removed {attr.attraction_name} due to opening hours constraint")
                            fixed_violations = True
            
            # 7. VALIDATE AND FIX OVERLAPS
            for day in range(repaired_solution.tourist.days):
                if day not in repaired_solution.tourist.locations or len(repaired_solution.tourist.locations[day]) <= 1:
                    continue
                
                time_slots = []
                for attr in repaired_solution.tourist.locations[day]:
                    if attr.attraction_name in repaired_solution.tourist.start_times[day]:
                        start_time = repaired_solution.tourist.start_times[day][attr.attraction_name]
                        end_time = start_time + attr.task_time
                        time_slots.append((attr, start_time, end_time))
                
                # Sort by start time
                time_slots.sort(key=lambda x: x[1])
                
                # Check for overlaps
                for i in range(len(time_slots) - 1):
                    curr_attr, curr_start, curr_end = time_slots[i]
                    next_attr, next_start, next_end = time_slots[i + 1]
                    
                    if curr_end > next_start:
                        print(f"   Overlapping visits on day {day + 1}: {curr_attr.attraction_name} and {next_attr.attraction_name}")
                        
                        # Try to adjust next start time
                        new_start = curr_end
                        if new_start + next_attr.task_time <= repaired_solution.tourist.touring_dict[day][1]:
                            repaired_solution.tourist.start_times[day][next_attr.attraction_name] = new_start
                            print(f"    Adjusted start time of {next_attr.attraction_name} to {new_start}")
                            fixed_violations = True
                        else:
                            # Remove next attraction if can't adjust
                            repaired_solution.tourist.remove(next_attr)
                            repaired_solution.unassigned.append(next_attr)
                            print(f"    Removed {next_attr.attraction_name} to resolve overlap")
                            fixed_violations = True
            
            # Update included categories after all modifications
            repaired_solution.update_included_categories()
            
            # If no violations were fixed this iteration, we're done
            if not fixed_violations:
                print(f"   All constraints validated successfully at iteration {iteration+1}!")
                break
            
            if iteration == max_iterations - 1:
                print("   Maximum iterations reached. Some constraints may still be violated.")

        # Calculate final objective
        repaired_solution.update_included_categories()
        sc_obj_neg = repaired_solution.objective()
        sc_obj_pos = -sc_obj_neg

        # Calculate impact on quality
        impact = (1 - sc_obj_pos / orig_obj_pos) * 100 if orig_obj_pos > 0 else 0

        print(f" Scenario {scenario['scenario']} objective: {sc_obj_pos:.4f}")
        print(f" Impact on quality: {impact:.1f}%")

        # Count total attractions
        attraction_count = sum(len(repaired_solution.tourist.locations.get(d, []))
                            for d in range(repaired_solution.tourist.days))

        # Record the results
        scenario_results.append({
            "scenario": scenario['scenario'],
            "objective": sc_obj_pos,
            "impact": impact,
            "attractions": attraction_count,
            "forecasts": scenario["forecasts"],
            "disrupted_days": scenario["disrupted_days"],
            "percent_of_best": sc_obj_pos / orig_obj_pos * 100 if orig_obj_pos > 0 else 0
        })

        # Store and save repaired solution
        repaired_solutions[scenario['scenario']] = repaired_solution

        # Save to JSON
        json_filename = f"disruption_routes/Tourist_{tourist_id}_Scenario_{scenario['scenario']}_quickrepair.json"
        save_solution_to_json(repaired_solution, json_filename, scenario['scenario'], scenario["forecasts"])
        print(f" Saved quick repair route to {json_filename}")

    return scenario_results, repaired_solutions


def alns_replan(solution, tourist_id, num_scenarios=3, seed=None):
    """
    Full ALNS replanning after disruptions for dynamic planning.
    Uses enhanced objective function that incorporates stability and proximity.

    Args:
        solution: Original SMJSP solution object
        tourist_id: Tourist ID for saving files
        num_scenarios: Number of disruption scenarios to simulate
        seed: Random seed for reproducibility

    Returns:
        A list of scenario result dictionaries and replanned solutions
    """
    if seed is not None:
        random.seed(seed)

    # Calculate original objective value
    orig_obj_neg = solution.objective()
    orig_obj_pos = -orig_obj_neg
    print("\nEvaluating full ALNS replanning with enhanced objective function...")
    print(f"Original solution objective (positive): {orig_obj_pos:.4f}")

    # Generate disruption scenarios
    scenarios = generate_disruption_scenarios(solution, num_scenarios=num_scenarios, seed=seed)
    scenario_results = []
    replanned_solutions = {}

    os.makedirs("disruption_routes", exist_ok=True)

    # Map disruption names to functions from disruptions.py
    disruption_map = {
        "normal": nothing_happens_attraction,
        "rainy": rainy_day_attraction,
        "heat_wave": heat_wave_attraction,
        "early_closure": early_closure,
        "family_day": family_day
    }

    for scenario in scenarios:
        print(f"\nScenario {scenario['scenario']}:")

        for d, forecast in scenario["forecasts"].items():
            print(f" Day {d + 1} forecast: {forecast.capitalize()}")
            
        # Store original plan before applying disruptions
        original_plan = {}
        for day in range(solution.tourist.days):
            if day in solution.tourist.locations:
                original_plan[day] = []
                for attr in solution.tourist.locations[day]:
                    if attr.attraction_name in solution.tourist.start_times[day]:
                        start_time = solution.tourist.start_times[day][attr.attraction_name]
                        location = (attr.lat_long[1], attr.lat_long[0])
                        original_plan[day].append((attr, start_time, location))

        # Apply disruptions with enhanced constraint handling
        disrupted_solution = apply_disruptions(solution, scenario["disruptions"])
        
        # Create enhanced objective SMJSP using the original solution and disrupted data
        enhanced_solution = create_enhanced_smjsp(
            original_solution=solution,
            tourist=disrupted_solution.tourist,
            attractions=disrupted_solution.attractions,
            stability_weight=0.3,
            proximity_weight=0.2
        )
        
        # Copy disrupted solution state to enhanced solution
        enhanced_solution.tourist = disrupted_solution.tourist
        enhanced_solution.unassigned = disrupted_solution.unassigned
        enhanced_solution.included_categories = disrupted_solution.included_categories
        
        # Store original plan for proximity calculations
        enhanced_solution.original_plan = original_plan

        # Setup ALNS
        alns_seed = seed + scenario['scenario'] if seed is not None else random.randint(0, 9999)
        random_state = rnd.RandomState(alns_seed)
        alns = ALNS(random_state)

        # Add destroy operators
        alns.add_destroy_operator(destroy_random, "Random")
        alns.add_destroy_operator(destroy_worst, "Worst")
        alns.add_destroy_operator(destroy_day, "Day")
        alns.add_destroy_operator(destroy_preference, "Preference")
        alns.add_destroy_operator(destroy_cluster, "Cluster")
        alns.add_destroy_operator(destroy_recently_added, "Recent")
        alns.add_destroy_operator(destroy_expensive, "Expensive")

        # Add repair operators
        alns.add_repair_operator(repair_random, "Random")
        alns.add_repair_operator(repair_greedy, "Greedy")
        alns.add_repair_operator(repair_regret, "Regret")
        alns.add_repair_operator(repair_nearest_neighbor, "Nearest")
        alns.add_repair_operator(repair_rainy, "Rainy")
        alns.add_repair_operator(repair_diversity_first, "DiversityFirst")
        alns.add_repair_operator(repair_time_slot_fit, "TimeSlotFit")
        alns.add_repair_operator(repair_cheapest_fun_proximity, "CheapestFunProx")
        alns.add_repair_operator(repair_maximize_attractions, "MaxAttr")
        alns.add_repair_operator(repair_balanced, "Balanced")
        alns.add_repair_operator(repair_minimize_changes, "MinimizeChanges")

        # ALNS parameters
        omegas = [15, 10, 8, 1]
        lambda_ = 0.8
        criterion = SimulatedAnnealing(500, 0.5, 0.9999, "exponential")

        # Run ALNS with validation
        print(f" Running ALNS replanning with enhanced objective for scenario {scenario['scenario']}...")
        result = alns.iterate(
            enhanced_solution,
            omegas,
            lambda_,
            criterion,
            iterations=5000,
            collect_stats=False
        )

        replanned_solution = result.best_state

        # Post-processing after ALNS to ensure all constraints are met
        print(" Performing post-processing to ensure all constraints are met...")

        # Apply stronger validation for disruption-specific constraints
        for day_str, forecast in scenario["forecasts"].items():
            day = int(day_str)
            forecast_type = forecast.lower()

            if day not in replanned_solution.tourist.locations:
                continue

            # Get list of attractions for this day
            attractions_for_today = list(replanned_solution.tourist.locations[day])
            attractions_to_remove = []

            for attr in attractions_for_today:
                # Apply specific constraints based on forecast type
                if forecast_type == "rainy" and "Outdoor" in attr.categories:
                    attractions_to_remove.append(attr)

                elif forecast_type == "heat_wave" and any(
                        cat in attr.categories for cat in ["Sporty", "Nature", "Outdoor"]):
                    attractions_to_remove.append(attr)

                elif forecast_type == "early_closure" and "Outdoor" not in attr.categories:
                    start_time = replanned_solution.tourist.start_times[day].get(attr.attraction_name)
                    if start_time is not None:
                        # For early closure, indoor attractions close at noon
                        # If start time is at or after noon, remove it
                        if start_time >= 12:
                            attractions_to_remove.append(attr)
                            print(
                                f"   Removing {attr.attraction_name} on day {day + 1}: starts at {start_time} (after noon)")
                        else:
                            # If it extends past noon, remove it
                            end_time = start_time + attr.task_time
                            if end_time > 12:
                                attractions_to_remove.append(attr)
                                print(
                                    f"   Removing {attr.attraction_name} on day {day + 1}: ends at {end_time} (after noon)")

                elif forecast_type == "family_day" and "Family" not in attr.categories:
                    attractions_to_remove.append(attr)

            # Remove all problematic attractions
            for attr in attractions_to_remove:
                replanned_solution.tourist.remove(attr)
                replanned_solution.unassigned.append(attr)
                print(f"   Removed {attr.attraction_name} from day {day + 1} due to {forecast_type} constraints")

        # Validate and fix break time constraints
        for day in range(replanned_solution.tourist.days):
            if day not in replanned_solution.tourist.locations or len(replanned_solution.tourist.locations[day]) <= 1:
                continue

            # Gather time slots for this day
            time_slots = []
            for attr in replanned_solution.tourist.locations[day]:
                if attr.attraction_name in replanned_solution.tourist.start_times[day]:
                    start_time = replanned_solution.tourist.start_times[day][attr.attraction_name]
                    end_time = start_time + attr.task_time
                    time_slots.append((attr, start_time, end_time))

            # Sort by start time
            time_slots.sort(key=lambda x: x[1])

            # Calculate total break time
            total_break_time = 0
            for i in range(len(time_slots) - 1):
                _, _, curr_end = time_slots[i]
                _, next_start, _ = time_slots[i + 1]
                break_time = next_start - curr_end
                total_break_time += break_time

            # If break time constraint is violated
            if total_break_time < 1:
                print(f"   Break time constraint violated on day {day + 1}: {total_break_time:.1f} hours (need 1 hour)")

                # Try to fix by removing the attraction with the smallest fun score
                if time_slots:
                    scored_attractions = [(attr, replanned_solution.get_attraction_fun_score(attr))
                                        for attr, _, _ in time_slots]
                    scored_attractions.sort(key=lambda x: x[1])  # Sort by ascending fun score

                    attr_to_remove = scored_attractions[0][0]
                    replanned_solution.tourist.remove(attr_to_remove)
                    replanned_solution.unassigned.append(attr_to_remove)
                    print(f"   Removed {attr_to_remove.attraction_name} to fix break time constraint")

        # Validate travel time constraints
        for day in range(replanned_solution.tourist.days):
            if day not in replanned_solution.tourist.locations or not replanned_solution.tourist.locations[day]:
                continue

            hotel_lat = replanned_solution.tourist.hotel_lat
            hotel_long = replanned_solution.tourist.hotel_long
            
            # Convert to list with start times
            activities = []
            for attr in replanned_solution.tourist.locations[day]:
                if attr.attraction_name in replanned_solution.tourist.start_times[day]:
                    start_time = replanned_solution.tourist.start_times[day][attr.attraction_name]
                    end_time = start_time + attr.task_time
                    activities.append((attr, start_time, end_time))

            activities.sort(key=lambda x: x[1])  # Sort by start time

            if not activities:
                continue

            # Check travel time from hotel to first attraction
            first_attr, first_start, _ = activities[0]
            distance = calculate_distance(
                hotel_lat, hotel_long,
                first_attr.lat_long[1], first_attr.lat_long[0]
            )
            travel_time = distance / 30.0  # 30 km/h average speed

            if first_start < replanned_solution.tourist.touring_dict[day][0] + travel_time:
                print(f"   Travel time constraint violated on day {day + 1}: Cannot reach first attraction in time")
                # Try to adjust start time
                new_start = replanned_solution.tourist.touring_dict[day][0] + travel_time
                if new_start + first_attr.task_time <= replanned_solution.tourist.touring_dict[day][1]:
                    replanned_solution.tourist.start_times[day][first_attr.attraction_name] = new_start
                    print(f"    Adjusted start time of {first_attr.attraction_name} to {new_start:.1f}")
                else:
                    # Remove attraction if can't adjust
                    replanned_solution.tourist.remove(first_attr)
                    replanned_solution.unassigned.append(first_attr)
                    print(f"    Removed {first_attr.attraction_name} due to insufficient travel time")
                    continue  # Skip to next day since we modified activities

            # Check travel time between attractions
            for i in range(len(activities) - 1):
                curr_attr, _, curr_end = activities[i]
                next_attr, next_start, _ = activities[i + 1]

                distance = calculate_distance(
                    curr_attr.lat_long[1], curr_attr.lat_long[0],
                    next_attr.lat_long[1], next_attr.lat_long[0]
                )
                travel_time = distance / 30.0

                if next_start < curr_end + travel_time:
                    print(
                        f"   Travel time constraint violated on day {day + 1}: Cannot travel between attractions in time")
                    # Try to adjust next attraction's start time
                    new_start = curr_end + travel_time
                    if new_start + next_attr.task_time <= replanned_solution.tourist.touring_dict[day][1]:
                        replanned_solution.tourist.start_times[day][next_attr.attraction_name] = new_start
                        print(f"    Adjusted start time of {next_attr.attraction_name} to {new_start:.1f}")
                    else:
                        # Remove next attraction if can't adjust
                        replanned_solution.tourist.remove(next_attr)
                        replanned_solution.unassigned.append(next_attr)
                        print(f"    Removed {next_attr.attraction_name} due to insufficient travel time")
                        break  # Exit loop since we modified activities

            # Reload activities since we might have modified the list
            activities = []
            for attr in replanned_solution.tourist.locations[day]:
                if attr.attraction_name in replanned_solution.tourist.start_times[day]:
                    start_time = replanned_solution.tourist.start_times[day][attr.attraction_name]
                    end_time = start_time + attr.task_time
                    activities.append((attr, start_time, end_time))

            activities.sort(key=lambda x: x[1])

            if not activities:
                continue

            # Check travel time from last attraction back to hotel
            last_attr, _, last_end = activities[-1]
            distance = calculate_distance(
                last_attr.lat_long[1], last_attr.lat_long[0],
                hotel_lat, hotel_long
            )
            travel_time = distance / 30.0

            if last_end + travel_time > replanned_solution.tourist.touring_dict[day][1]:
                print(f"   Travel time constraint violated on day {day + 1}: Cannot return to hotel in time")
                # Since this is the last activity, we just remove it
                replanned_solution.tourist.remove(last_attr)
                replanned_solution.unassigned.append(last_attr)
                print(f"    Removed {last_attr.attraction_name} due to insufficient travel time back to hotel")

        # Update included categories for the solution
        replanned_solution.update_included_categories()

        # Calculate final objective
        final_obj_neg, breakdown = replanned_solution.objective(return_breakdown=True)
        final_obj_pos = -final_obj_neg
        impact = (1 - final_obj_pos / orig_obj_pos) * 100 if orig_obj_pos > 0 else 0

        # Print out enhanced objective components if available
        if "stability_score" in breakdown:
            print(f" Stability score: {breakdown['stability_score']:.4f}")
            print(f" Proximity score: {breakdown['proximity_score']:.4f}")
            if "schedule_similarity" in breakdown:
                print(f" Schedule similarity: {breakdown['schedule_similarity']:.4f}")

        print(f" Scenario {scenario['scenario']} objective: {final_obj_pos:.4f}")
        print(f" Impact on quality: {impact:.1f}%")

        # Count attractions
        attraction_count = sum(len(replanned_solution.tourist.locations.get(d, []))
                            for d in range(replanned_solution.tourist.days))

        # Record results
        scenario_results.append({
            "scenario": scenario['scenario'],
            "objective": final_obj_pos,
            "impact": impact,
            "attractions": attraction_count,
            "forecasts": scenario["forecasts"],
            "disrupted_days": scenario["disrupted_days"],
            "percent_of_best": final_obj_pos / orig_obj_pos * 100,
            "stability_score": breakdown.get("stability_score", 0),
            "proximity_score": breakdown.get("proximity_score", 0)
        })

        # Store and save replanned solution
        replanned_solutions[scenario['scenario']] = replanned_solution

        # Save to JSON
        json_filename = f"disruption_routes/Tourist_{tourist_id}_Scenario_{scenario['scenario']}_replanned.json"
        save_solution_to_json(replanned_solution, json_filename, scenario['scenario'], scenario["forecasts"])
        print(f" Saved replanned route to {json_filename}")

    return scenario_results, replanned_solutions


def count_total_attractions(solution):
    """Count total attractions in a solution."""
    return sum(len(solution.tourist.locations.get(d, [])) for d in range(solution.tourist.days))

def run_disruption_comparison(tourist, attractions, output_dir="stage2_results", seed=123):
    """
    Run comprehensive comparison of planning methods under disruptions.

    Args:
        tourist: Tourist object
        attractions: List of attractions
        output_dir: Directory to save results
        seed: Random seed

    Returns:
        Dictionary with comparison results
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # Record basic info
    results["tourist_id"] = tourist.idx
    results["preferences"] = tourist.preferences
    results["days"] = tourist.days
    results["budget"] = tourist.budget

    # ---------- STAGE 1: GENERATE SOLUTIONS ----------

    # Load initial solution instead of regenerating
    initial_sol_path = f"Tourist_{tourist.idx}_initial_solution.json"

    if os.path.exists(initial_sol_path):
        print(f"\nLoading initial solution from {initial_sol_path}...")
        initial_solution = SMJSP.load_from_file(initial_sol_path, attractions)
        # Use standardized metrics
        initial_metrics = calculate_objective_metrics(initial_solution)

    else:
        print(f"\nCould not find saved initial solution. Generating new one...")

        # Fallback to generating a new solution
        initial_smjsp = SMJSP(
            tourist,
            attractions,
            weighting=w_norm,
            diversity_bonus=args.diversity_bonus
        )

        initial_smjsp.random_initialize(seed)
        initial_solution = initial_smjsp.copy()
        initial_metrics = calculate_objective_metrics(initial_solution)


    # Load optimized solution instead of regenerating
    optimized_sol_path = os.path.join("./stage1_results", f"Tourist_{tourist.idx}_optimized_solution.json")

    if os.path.exists(optimized_sol_path):
        print(f"\nLoading optimized solution from {optimized_sol_path}...")
        optimized_solution = SMJSP.load_from_file(optimized_sol_path, attractions)
        optimized_metrics = calculate_objective_metrics(optimized_solution)

    else:
        print("\nCould not find saved optimized solution. Running ALNS optimization...")

        # Fallback to running ALNS
        alns = ALNS(rnd.RandomState(seed))

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

        result = alns.iterate(
            initial_solution,
            [5, 3, 2, 1],
            0.8,
            SimulatedAnnealing(200, 0.5, 0.9999, "exponential"),
            iterations=5000
        )

        optimized_solution = result.best_state
        optimized_metrics = calculate_objective_metrics(optimized_solution)

    # Use standardized comparison
    stage1_comparison = compare_solutions(
        initial_metrics, 
        optimized_metrics,
        "Initial", 
        "Optimized"
    )

    # Store results with standardized metrics
    results["initial"] = {
        "solution": initial_solution,
        "metrics": initial_metrics
    }

    results["optimized"] = {
        "solution": optimized_solution,
        "metrics": optimized_metrics,
        "comparison": stage1_comparison
    }

    print(f"Initial solution objective: {initial_metrics['objective_positive']:.4f}")
    print(f"Optimized solution objective: {optimized_metrics['objective_positive']:.4f}")
    print(f"Improvement: {stage1_comparison['improvement_description']}")

    # ---------- STAGE 2: DISRUPTION COMPARISON ----------

    # Prepare methods to compare - Updated to include both replanning methods
    methods = {
        "initial": {"solution": initial_solution, "metrics": initial_metrics},
        "optimized": {"solution": optimized_solution, "metrics": optimized_metrics},
        "quick_repair": {"solution": initial_solution, "metrics": initial_metrics},
        "alns_replan": {"solution": initial_solution, "metrics": initial_metrics}
    }


    # Process each method with standardized metrics
    results["disruption_comparison"] = []

    for method_name, method_data in methods.items():
        print(f"\nTesting {method_name} method:")
        method_solution = method_data["solution"]

        # Apply different processing based on method
        if method_name == "quick_repair":
            # Use quick_repair function
            disruption_scenarios, qr_solutions = quick_repair(
                method_solution,
                tourist.idx,
                num_scenarios=3,
                seed=seed
            )

            # Calculate standardized metrics for each repaired solution
            for scenario_id, repaired_solution in qr_solutions.items():
                scenario_metrics = calculate_objective_metrics(repaired_solution)
                # Update scenario data with standardized metrics
                for scenario in disruption_scenarios:
                    if scenario["scenario"] == scenario_id:
                        scenario["metrics"] = scenario_metrics
                        scenario["objective"] = scenario_metrics["objective_positive"]
                        break

            # Store repaired solutions with metrics
            results["quick_repair_solutions"] = {
                scenario_id: {
                    "solution": solution,
                    "metrics": calculate_objective_metrics(solution)
                } for scenario_id, solution in qr_solutions.items()
            }

        elif method_name == "alns_replan":
            # Use ALNS replanning
            disruption_scenarios, ar_solutions = alns_replan(
                method_solution,
                tourist.idx,
                num_scenarios=3,
                seed=seed
            )

            # Calculate standardized metrics for each replanned solution
            for scenario_id, replanned_solution in ar_solutions.items():
                scenario_metrics = calculate_objective_metrics(replanned_solution)
                # Update scenario data with standardized metrics
                for scenario in disruption_scenarios:
                    if scenario["scenario"] == scenario_id:
                        scenario["metrics"] = scenario_metrics
                        scenario["objective"] = scenario_metrics["objective_positive"]
                        break

            # Store replanned solutions with metrics
            results["alns_replan_solutions"] = {
                scenario_id: {
                    "solution": solution,
                    "metrics": calculate_objective_metrics(solution)
                } for scenario_id, solution in ar_solutions.items()
            }

        else:
            # Generate common disruption scenarios
            scenarios = generate_disruption_scenarios(method_solution, num_scenarios=3, seed=seed)
            disruption_scenarios = []

            for scenario in scenarios:
                # Apply disruptions
                disrupted_solution = apply_disruptions(method_solution, scenario["disruptions"])

                # Use standardized metrics
                scenario_metrics = calculate_objective_metrics(disrupted_solution)
                start_metrics = method_data["metrics"]

                # Calculate impact using standardized comparison
                scenario_comparison = compare_solutions(
                    start_metrics,
                    scenario_metrics,
                    f"{method_name} Original",
                    f"{method_name} Disrupted"
                )

                # Add scenario result with standardized metrics
                disruption_scenarios.append({
                    "scenario": scenario['scenario'],
                    "objective": scenario_metrics["objective_positive"],
                    "metrics": scenario_metrics,
                    "impact": scenario_comparison["improvement_percentage"] * -1,  # Convert to impact (negative of improvement)
                    "attractions": scenario_metrics["attractions_count"],
                    "forecasts": scenario["forecasts"],
                    "disrupted_days": scenario["disrupted_days"],
                    "percent_of_best": scenario_metrics["objective_positive"] / start_metrics["objective_positive"] * 100 
                        if start_metrics["objective_positive"] > 0 else 0
                })

        # Add method results to overall results with standardized metrics
        results["disruption_comparison"].append({
            "method": method_name,
            "scenarios": disruption_scenarios
        })

    # Summarize results
    print("\n" + "=" * 60)
    print("DISRUPTION HANDLING COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Method':<15} {'Avg Obj':<10} {'Avg Attractions':<15}")
    print("-" * 70)

    for method_results in results["disruption_comparison"]:
        method_name = method_results["method"]
        scenarios = method_results["scenarios"]

        # Use better display name for methods
        display_name = method_name
        if method_name == "quick_repair":
            display_name = "Quick Repair"
        elif method_name == "alns_replan":
            display_name = "ALNS Replan"

        # Calculate averages using standardized metrics
        avg_obj = sum(s["objective"] for s in scenarios) / len(scenarios) if scenarios else 0
        avg_impact = sum(s["impact"] for s in scenarios) / len(scenarios) if scenarios else 0
        avg_attractions = sum(s["attractions"] for s in scenarios) / len(scenarios) if scenarios else 0


        print(f"{display_name:<15} {avg_obj:.4f}   {avg_attractions:.1f}")

    # Save detailed results
    save_disruption_results(results, os.path.join(output_dir, f"tourist_{tourist.idx}_disruption_comparison.txt"))

    return results


def save_disruption_results(results, filename):
    """
    Save disruption comparison results to a file.

    Args:
        results: Dictionary with comparison results
        filename: Output filename
    """
    lines = []

    # Basic info
    lines.append("DISRUPTION HANDLING COMPARISON (Objective Values)")
    lines.append("=" * 60)
    lines.append(f"Tourist ID: {results['tourist_id']}")
    lines.append(f"Preferences: {results['preferences']}")
    lines.append(f"Days: {results['days']}")
    lines.append(f"Budget: SGD ${results['budget']}")
    lines.append("")

    # Stage 1 results (no disruptions)
    lines.append("STAGE 1: PLANNING METHODS (No Disruptions)")
    lines.append("-" * 60)

    initial_obj = results["initial"]["metrics"]["objective_positive"]
    optimized_obj = results["optimized"]["metrics"]["objective_positive"]

    lines.append(f"Initial Heuristic Objective: {initial_obj:.4f}")
    lines.append(f"ALNS Optimized Objective: {optimized_obj:.4f}")
    lines.append(f"Improvement: {results['optimized']['comparison']['improvement_description']}")
    lines.append("")

    # Stage 2: Disruption comparison focusing on objective values
    lines.append("STAGE 2: Disruption Handling Comparison (Objectives)")
    lines.append("-" * 90)  # Wider line to accommodate disruption details
    
    for method_result in results["disruption_comparison"]:
        method_name = method_result["method"]
        # Use better display name for methods
        if method_name == "quick_repair":
            display_name = "Quick Repair"
        elif method_name == "alns_replan":
            display_name = "ALNS Replanning"
        else:
            display_name = method_name.capitalize()

        lines.append(f"\nMethod: {display_name}")
        
        # Add appropriate header based on method
        if method_name == "alns_replan":
            lines.append("Scenario    Objective      Stability    Proximity    Disruptions")
        else:
            lines.append("Scenario    Objective      Disruptions")
        
        lines.append("-" * 90)

        # Process scenarios with appropriate formatting
        for scenario in method_result["scenarios"]:
            scenario_number = scenario.get("scenario", "N/A")
            obj_value = scenario.get("objective", 0)
            
            # Get disruption info
            disruption_summary = ""
            if "forecasts" in scenario:
                day_disruptions = []
                for day, forecast in scenario["forecasts"].items():
                    if forecast != "normal":  # Only include non-normal days
                        day_disruptions.append(f"Day {int(day) + 1}: {forecast.capitalize()}")
                
                disruption_summary = ", ".join(day_disruptions) if day_disruptions else "No disruptions"
            
            # Format output line based on method
            if method_name == "alns_replan":
                stability = scenario.get("stability_score", 0)
                proximity = scenario.get("proximity_score", 0)
                lines.append(f"{scenario_number:<10}  {obj_value:<10.4f}  {stability:<10.4f}  {proximity:<10.4f}  {disruption_summary}")
            else:
                lines.append(f"{scenario_number:<10}  {obj_value:<10.4f}  {disruption_summary}")

    # Overall summary: average objective per method
    lines.append("\n" + "=" * 60)
    lines.append("OVERALL AVERAGE OBJECTIVE VALUES")
    lines.append("=" * 60)
    lines.append(f"{'Method':<18} {'Avg Objective':<15} {'Avg Attractions':<15}")
    lines.append("-" * 70)
    
    # Calculate method averages using standardized metrics
    method_averages = {}
    
    for method_result in results["disruption_comparison"]:
        method_name = method_result["method"]
        # Use better display name for methods
        if method_name == "quick_repair":
            display_name = "Quick Repair"
        elif method_name == "alns_replan":
            display_name = "ALNS Replanning"
        else:
            display_name = method_name.capitalize()

        scenarios = method_result["scenarios"]
        if scenarios:
            avg_obj = sum(s["objective"] for s in scenarios) / len(scenarios)
            avg_impact = sum(s["impact"] for s in scenarios) / len(scenarios)
            avg_attractions = sum(s["attractions"] for s in scenarios) / len(scenarios)
            
            method_averages[method_name] = {
                "avg_obj": avg_obj,
                "avg_impact": avg_impact,
                "avg_attractions": avg_attractions
            }
            
            lines.append(f"{display_name:<18} {avg_obj:<15.4f} {avg_attractions:<15.1f}")
        else:
            lines.append(f"{display_name:<18} No data")

    # Compare replanning methods using standardized comparison
    quick_repair_data = method_averages.get("quick_repair")
    alns_replan_data = method_averages.get("alns_replan")

    if quick_repair_data and alns_replan_data:
        lines.append("\nREPLANNING METHOD COMPARISON:")
        
        # Use standardized comparison logic
        qr_metrics = {"objective_positive": quick_repair_data["avg_obj"]}
        alns_metrics = {"objective_positive": alns_replan_data["avg_obj"]}
        
        method_comparison = compare_solutions(
            qr_metrics, 
            alns_metrics,
            "Quick Repair", 
            "ALNS Replanning"
        )

        # Report comparison results
        better_method = method_comparison["better_solution"]
        improvement_desc = method_comparison["improvement_description"]
        lines.append(f"- {better_method} achieves better objective values: {improvement_desc}")

        # Impact difference
        impact_diff = quick_repair_data["avg_impact"] - alns_replan_data["avg_impact"]
        better_impact = "ALNS Replanning" if impact_diff > 0 else "Quick Repair"
        if abs(impact_diff) > 1:
            lines.append(f"- {better_impact} has {abs(impact_diff):.1f}% less disruption impact")
        else:
            lines.append("- Both methods handle disruptions similarly")

        # Attraction count difference
        attr_diff_pct = ((alns_replan_data["avg_attractions"] / quick_repair_data["avg_attractions"]) - 1) * 100 if quick_repair_data["avg_attractions"] > 0 else 0
        if abs(attr_diff_pct) > 5:
            better_attr = "ALNS Replanning" if attr_diff_pct > 0 else "Quick Repair"
            lines.append(f"- {better_attr} includes {abs(attr_diff_pct):.1f}% more attractions")
        else:
            lines.append("- Both methods include a similar number of attractions")

    # Add information about replanned routes
    lines.append("\nReplanned Routes Information")
    lines.append("-" * 60)
    lines.append(f"Detailed replanned routes have been saved to the 'disruption_routes' directory:")
    lines.append(f"Files:")
    lines.append(f"  - Tourist_{results['tourist_id']}_Scenario_X_replanned.json (ALNS Replanning)")
    lines.append(f"  - Tourist_{results['tourist_id']}_Scenario_X_quickrepair.json (Quick Repair)")
    lines.append(f"  where X is the scenario number")

    # Write to file
    output = "\n".join(lines)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"\nDetailed results saved to: {filename}")


def aggregate_results(all_tourist_results):
    """
    Aggregate results across all tourists.

    Args:
        all_tourist_results: List of results dictionaries for multiple tourists

    Returns:
        Dictionary with aggregated statistics
    """
    aggregated = {}

    # Initialize aggregated structure
    for method in ["initial", "optimized", "quick_repair", "alns_replan"]:
        aggregated[method] = {"objectives": [], "impacts": [], "attractions": []}

    # Collect data
    for tourist_results in all_tourist_results:
        # Handle initial and optimized solutions
        if "initial" in tourist_results and "objective" in tourist_results["initial"]:
            aggregated["initial"]["objectives"].append(tourist_results["initial"]["objective"])

        if "optimized" in tourist_results and "objective" in tourist_results["optimized"]:
            aggregated["optimized"]["objectives"].append(tourist_results["optimized"]["objective"])

        # Handle disruption comparison data
        if "disruption_comparison" in tourist_results:
            for method_result in tourist_results["disruption_comparison"]:
                method = method_result["method"]
                if method in ["quick_repair", "alns_replan"] and method_result["scenarios"]:
                    # Average over scenarios
                    objectives = [s["objective"] for s in method_result["scenarios"]]
                    impacts = [s["impact"] for s in method_result["scenarios"]]
                    attractions = [s["attractions"] for s in method_result["scenarios"]]

                    aggregated[method]["objectives"].extend(objectives)
                    aggregated[method]["impacts"].extend(impacts)
                    aggregated[method]["attractions"].extend(attractions)

    # Calculate averages
    averages = {}
    for method, data in aggregated.items():
        averages[method] = {
            "avg_objective": sum(data["objectives"]) / len(data["objectives"]) if data["objectives"] else 0,
        }
        if "impacts" in data and data["impacts"]:
            averages[method]["avg_impact"] = sum(data["impacts"]) / len(data["impacts"])
            averages[method]["avg_attractions"] = sum(data["attractions"]) / len(data["attractions"])
        else:
            averages[method]["avg_impact"] = 0
            averages[method]["avg_attractions"] = 0

    return averages


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Compare planning methods under disruptions")
    parser.add_argument("attraction_csv", type=str, help="Path to attraction data CSV")
    parser.add_argument("tourist_csv", type=str, help="Path to tourist data CSV")
    parser.add_argument("--tourist_id", type=int, default=None, help="Specific tourist ID to evaluate")
    parser.add_argument("--output_dir", type=str, default="stage2_results", help="Directory to save results")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "aggregate"],
                        help="Specify mode: 'single' for single tourist or 'aggregate' for all tourists.")

    parser.add_argument("--diversity_bonus", type=float, default=15,
                        help="bonus points for first attraction of each category")
    parser.add_argument("--fun_weight", type=float, default=0.6,
                        help="weight for fun score in objective function")  # will be normalized later
    parser.add_argument("--distance_weight", type=float, default=0.3,
                        help="weight for distance in objective function")  # Decreased from 0.4, will be normalized later
    parser.add_argument("--attraction_weight", type=float, default=0.5,
                        help="weight for number of attractions in objective function")  # Increased from 0.3, will be normalized later
    args = parser.parse_args()

    # Weights
    w_list = [args.fun_weight, args.distance_weight, args.attraction_weight]
    s_ = sum(w_list)
    if s_ > 0:
        w_norm = [v / s_ for v in w_list]
    else:
        w_norm = w_list

    print("\n" + "=" * 70)
    print(" Tourism Itinerary Planning under Disruptions - Stage 2 Comparison")
    print("=" * 70)
    print(f"Comparing replanning methods: Quick Repair vs ALNS Replanning")
    print(
        f"Objective weights: Fun={args.fun_weight}, Distance={args.distance_weight}, Attractions={args.attraction_weight}")
    print(f"Diversity bonus: {args.diversity_bonus} points")

    # Parse data
    parsed = Parser(args.attraction_csv, args.tourist_csv)

    # Adjust day indexing
    for a in parsed.attractions:
        new_opening = {}
        for dayk, hourset in a.opening_hours.items():
            new_opening[dayk - 1] = hourset
        a.opening_hours = new_opening

    if args.mode == "single":
        # Select tourist
        if args.tourist_id is not None:
            tourist = next((t for t in parsed.tourists if int(t.idx) == args.tourist_id), None)
            if not tourist:
                print(f"Tourist ID {args.tourist_id} not found, abort.")
                sys.exit(1)
        else:
            random.seed(args.seed)
            tourist = random.choice(parsed.tourists)

        print(f"\nRunning disruption comparison for Tourist ID: {tourist.idx}")
        print(f"Preferences: {tourist.preferences}")
        print(f"Budget: SGD ${tourist.budget}")
        print(f"Days: {tourist.days}")

        # Run comparison
        results = run_disruption_comparison(
            tourist,
            parsed.attractions,
            output_dir=args.output_dir,
            seed=args.seed
        )

        print("\nDISRUPTION HANDLING RESULTS SUMMARY:")
        print("-" * 50)
        print("Tested both ALNS Replanning and Quick Repair methods under identical disruption scenarios")
        print(f"Results saved to {args.output_dir}/tourist_{tourist.idx}_disruption_comparison.txt")
        print(f"Detailed routes saved to disruption_routes/ directory")
        print(f"Comprehensive evaluation: Tourist_{tourist.idx}_Dynamic_Tour_Planner_comprehensive_evaluation.txt")
    elif args.mode == "aggregate":
        # Run for all tourists and aggregate
        aggregated_results = []
        for tourist in parsed.tourists:
            print(f"\n--- Processing Tourist ID: {tourist.idx} ---")
            results = run_disruption_comparison(
                tourist,
                parsed.attractions,
                output_dir=args.output_dir,
                seed=args.seed
            )
            aggregated_results.append(results)

        # Process and summarize the aggregated results
        total_initial_objs = []
        total_optimized_objs = []
        total_alns_replan_objs = []
        total_quick_repair_objs = []

        for r in aggregated_results:
            # Add initial and optimized objectives
            if "initial" in r and "objective" in r["initial"]:
                total_initial_objs.append(r["initial"]["objective"])

            if "optimized" in r and "objective" in r["optimized"]:
                total_optimized_objs.append(r["optimized"]["objective"])

            # Process replanning scenarios
            if "disruption_comparison" in r:
                for method_result in r["disruption_comparison"]:
                    if method_result["method"] == "alns_replan" and method_result["scenarios"]:
                        avg_obj = sum(s["objective"] for s in method_result["scenarios"]) / len(
                            method_result["scenarios"])
                        total_alns_replan_objs.append(avg_obj)
                    elif method_result["method"] == "quick_repair" and method_result["scenarios"]:
                        avg_obj = sum(s["objective"] for s in method_result["scenarios"]) / len(
                            method_result["scenarios"])
                        total_quick_repair_objs.append(avg_obj)

        # Calculate and print averages
        print("\nAGGREGATE RESULTS ACROSS ALL TOURISTS")
        print("-" * 50)

        if total_initial_objs:
            print(f"Average initial objectives: {sum(total_initial_objs) / len(total_initial_objs):.4f}")

        if total_optimized_objs:
            print(f"Average optimized objectives: {sum(total_optimized_objs) / len(total_optimized_objs):.4f}")

        if total_quick_repair_objs:
            print(f"Average Quick Repair objectives: {sum(total_quick_repair_objs) / len(total_quick_repair_objs):.4f}")

        if total_alns_replan_objs:
            print(
                f"Average ALNS Replanning objectives: {sum(total_alns_replan_objs) / len(total_alns_replan_objs):.4f}")

        # Compare the two replanning methods if both have data
        if total_quick_repair_objs and total_alns_replan_objs:
            qr_avg = sum(total_quick_repair_objs) / len(total_quick_repair_objs)
            ar_avg = sum(total_alns_replan_objs) / len(total_alns_replan_objs)

            diff_pct = (ar_avg - qr_avg) / qr_avg * 100 if qr_avg != 0 else 0
            better = "ALNS Replanning" if diff_pct > 0 else "Quick Repair"

            print("\nREPLANNING METHOD COMPARISON:")
            print(f"{better} achieves {abs(diff_pct):.2f}% better objective values on average across all tourists")