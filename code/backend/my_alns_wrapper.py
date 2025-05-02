import os
import sys
import numpy as np

# Adjust your base directory if needed
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

# Stage 1 code that can parse user input, run an ALNS baseline
from alns_main_stage1_UI import run_custom_tourist_from_input, save_smjsp_output_json

# Stage 2 code that applies disruptions, quick repair, or ALNS replan
from disruption_main_stage2 import apply_disruptions, quick_repair, alns_replan


def run_alns_with_params(params):
    """
    A combined Stage 1 + Stage 2 wrapper.

    1) Build and optimize a baseline itinerary via run_custom_tourist_from_input (Stage 1).
    2) If disruptions are specified, apply them and re-run either quick_repair or alns_replan (Stage 2).
    3) Return the final itinerary as a JSON/dict structure for your React front end.
    """

    print("[ALNS] Received frontend request with user parameters:", params)
    sys.stdout.flush()

    # -------------------------
    # 1. Stage 1 Baseline
    # -------------------------
    # This function does random_initialize + ALNS for the user’s days, budget, etc.
    baseline_solution = run_custom_tourist_from_input(
        params, seed=params.get("seed", 123)
    )

    # -------------------------
    # 2. Stage 2 Disruptions?
    # -------------------------
    disruptions_dict = params.get("disruptions", {})   # e.g. {0:"rainy",1:"heat_wave"}
    replan_method = params.get("replan_method", "quick")

    if not disruptions_dict:
        print("[ALNS] No disruptions => final = baseline")
        final_solution = baseline_solution

    else:
        print("[ALNS] Disruptions found:", disruptions_dict)
        print("[ALNS] Replan method:", replan_method)

        # Convert {dayIndex: forecast_type} into a list of (dayIndex, forecast_type)
        disruptions_list = []
        for (day_str, forecast_type) in disruptions_dict.items():
            day_idx = int(day_str)
            disruptions_list.append((day_idx, forecast_type))

        # Apply disruptions to the baseline solution
        disrupted_solution = apply_disruptions(baseline_solution, disruptions_list)

        # Now re-optimize with either quick_repair or alns_replan
        if replan_method == "quick":
            # quick_repair returns (scenario_results, dict_of_solutions)
            scenario_results, repaired_solutions = quick_repair(
                disrupted_solution,
                tourist_id=baseline_solution.tourist.idx,
                num_scenarios=1,
                seed=params.get("seed", 123),
            )
            # The quick_repair solutions are keyed by scenario index => scenario=0
            final_solution = repaired_solutions[0]

        else:
            # alns_replan returns (scenario_results, dict_of_solutions)
            scenario_results, replanned_solutions = alns_replan(
                disrupted_solution,
                tourist_id=baseline_solution.tourist.idx,
                num_scenarios=1,
                seed=params.get("seed", 123),
            )
            final_solution = replanned_solutions[0]

        print(f"[ALNS] Done Stage 2 replan. final objective={final_solution.objective():.4f}")

    # -------------------------
    # 3. Save + Return JSON
    # -------------------------
    # Save final solution to solution.json for debugging
    save_smjsp_output_json(final_solution)

    # Build the dictionary your React front end needs
    response = {
        "tourist": {
            "id": final_solution.tourist.idx,
            "budget": final_solution.tourist.budget,
            "hotelLat": final_solution.tourist.hotel_lat,
            "hotelLng": final_solution.tourist.hotel_long
        },
        "itinerary": []
    }

    # Convert days and attractions to the “day: X, attractions: []” structure
    for d in range(final_solution.tourist.days):
        day_entry = {"day": d + 1, "attractions": []}
        if d in final_solution.tourist.locations:
            for attr in final_solution.tourist.locations[d]:
                start_time = final_solution.tourist.start_times[d].get(attr.attraction_name, 0.0)
                end_time = start_time + attr.task_time
                day_entry["attractions"].append({
                    "name": attr.attraction_name,
                    "time": f"{start_time:.1f}-{end_time:.1f}",
                    "cost": attr.cost,
                    "categories": attr.categories,
                    "lat": attr.lat_long[1],
                    "lng": attr.lat_long[0],
                })

        response["itinerary"].append(day_entry)

    return response
