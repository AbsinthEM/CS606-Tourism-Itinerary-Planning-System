import copy
import random
import math
import numpy as np
from utils import calculate_distance

from rcjsp import SMJSP


### Destroy operators ###
def destroy_random(current, random_state, destroy_factor=0.5):
    """
    Randomly removes a percentage of attractions from the current solution.
    
    Args:
        current: SMJSP object 
        random_state: numpy.random.RandomState
        destroy_factor: percentage of attractions to remove (0-1)
    
    Returns:
        destroyed: SMJSP object after removing attractions
    """
    destroyed = current.copy()
    
    # Get all assigned attractions
    assigned_attractions = []
    for day in destroyed.tourist.locations:
        assigned_attractions.extend(destroyed.tourist.locations[day])
    
    if not assigned_attractions:
        return destroyed
    
    # Calculate number of attractions to remove
    num_to_remove = max(1, int(len(assigned_attractions) * destroy_factor))
    
    # Randomly select attractions to remove
    to_remove = random_state.choice(assigned_attractions, 
                                    size=min(num_to_remove, len(assigned_attractions)), 
                                    replace=False)
    # print(f"Removing {num_to_remove} attractions out of {len(assigned_attractions)}")
    # print(f"Before: {len(current.tourist.locations.get(0, []))} attractions on day 1")
    
    # Remove selected attractions
    for attraction in to_remove:
        destroyed.tourist.remove(attraction)
        destroyed.unassigned.append(attraction)
    
    # After removals:
    # print(f"After: {len(destroyed.tourist.locations.get(0, []))} attractions on day 1")

    
    return destroyed

def destroy_worst(current, random_state, destroy_factor=0.7):
    """
    Removes attractions with lowest contribution to the objective function.
    
    Args:
        current: SMJSP object
        random_state: numpy.random.RandomState
        destroy_factor: percentage of attractions to remove (0-1)
    
    Returns:
        destroyed: SMJSP object after removing attractions
    """
    destroyed = current.copy()
    
    # Get all assigned attractions
    assigned_attractions = []
    for day in destroyed.tourist.locations:
        # Only consider attractions that have valid start times
        for attraction in destroyed.tourist.locations[day]:
            if attraction.attraction_name in destroyed.tourist.start_times.get(day, {}):
                assigned_attractions.append(attraction)
    
    if not assigned_attractions:
        return destroyed
    
    # Calculate number of attractions to remove
    num_to_remove = max(1, int(len(assigned_attractions) * destroy_factor))
    
    # Calculate contribution of each attraction to objective
    contributions = []
    for attraction in assigned_attractions:
        try:
            # Create a copy without this attraction
            temp = destroyed.copy()
            temp.tourist.remove(attraction)
            temp.unassigned.append(attraction)
            
            # Calculate difference in objective
            origin_objective = destroyed.objective()
            new_objective = temp.objective()
            contribution = origin_objective - new_objective
            contributions.append((attraction, contribution))
        except Exception as e:
            # If there's an error, use a random contribution to avoid breaking the algorithm
            contributions.append((attraction, random_state.random()))
    
    # Sort by contribution (ascending)
    contributions.sort(key=lambda x: x[1])
    
    # Remove worst contributors
    for i in range(min(num_to_remove, len(contributions))):
        attraction = contributions[i][0]
        destroyed.tourist.remove(attraction)
        destroyed.unassigned.append(attraction)
    
    return destroyed

def destroy_day(current, random_state, destroy_factor=0.5):
    """
    Removes all attractions from a random day.
    
    Args:
        current: SMJSP object
        random_state: numpy.random.RandomState
        destroy_factor: not used, included for compatibility
    
    Returns:
        destroyed: SMJSP object after removing attractions
    """
    destroyed = current.copy()
    
    # Get days with assigned attractions
    days_with_attractions = [day for day in range(destroyed.tourist.days) 
                            if day in destroyed.tourist.locations and destroyed.tourist.locations[day]]
    # print("Days with attr ", days_with_attractions)
    
    if not days_with_attractions:
        return destroyed
    
    # Select a random day
    day_to_clear = random_state.choice(days_with_attractions)
    
    # Remove all attractions from that day
    attractions_to_remove = destroyed.tourist.locations[day_to_clear].copy()
    for attraction in attractions_to_remove:
        destroyed.tourist.remove(attraction)
        destroyed.unassigned.append(attraction)
    
    return destroyed

def destroy_preference(current, random_state, destroy_factor=0.5):
    """
    Removes attractions that least match the tourist's preferences based on fun score.
    
    Args:
        current: SMJSP object
        random_state: numpy.random.RandomState
        destroy_factor: percentage of attractions to remove (0-1)
    
    Returns:
        destroyed: SMJSP object after removing attractions
    """
    destroyed = current.copy()
    
    # Get all assigned attractions
    assigned_attractions = []
    for day in destroyed.tourist.locations:
        assigned_attractions.extend(destroyed.tourist.locations[day])
    
    if not assigned_attractions:
        return destroyed
    
    # Calculate number of attractions to remove
    num_to_remove = max(1, int(len(assigned_attractions) * destroy_factor))
    
    # Make sure included_categories is up to date
    destroyed.update_included_categories()
    
    # Score each attraction by fun score (including diversity bonus = False)
    # We don't want to include diversity bonus here because we're evaluating
    # each attraction in isolation, not considering new categories
    fun_scores = []
    for attraction in assigned_attractions:
        # Use include_diversity_bonus=False to focus only on preference match
        score = destroyed.get_attraction_fun_score(attraction, include_diversity_bonus=False)
        # Add some randomness for tie breaking
        fun_scores.append((attraction, score + random_state.random() * 0.1))
    
    # Sort by score (ascending)
    fun_scores.sort(key=lambda x: x[1])
    
    # Remove worst preference matches
    for i in range(min(num_to_remove, len(fun_scores))):
        attraction = fun_scores[i][0]
        destroyed.tourist.remove(attraction)
        destroyed.unassigned.append(attraction)
    
    # Update included categories after removals
    destroyed.update_included_categories()
    
    return destroyed
def destroy_cluster(current, random_state, destroy_factor=0.5):
    """
    Removes a cluster of geographically close attractions.
    
    Args:
        current: SMJSP object
        random_state: numpy.random.RandomState
        destroy_factor: percentage of attractions to remove (0-1)
    
    Returns:
        destroyed: SMJSP object after removing attractions
    """
    destroyed = current.copy()
    
    # Get all assigned attractions
    assigned_attractions = []
    for day in destroyed.tourist.locations:
        assigned_attractions.extend(destroyed.tourist.locations[day])
    
    if not assigned_attractions:
        return destroyed
    
    # Calculate number of attractions to remove
    num_to_remove = max(1, int(len(assigned_attractions) * destroy_factor))
    
    # Randomly select a seed attraction
    if len(assigned_attractions) > 0:
        seed_idx = random_state.randint(0, len(assigned_attractions))
        seed_attraction = assigned_attractions[min(seed_idx, len(assigned_attractions)-1)]
        
        # Calculate distances from seed to all other attractions
        distances = []
        for attraction in assigned_attractions:
            if attraction != seed_attraction:
                dist = calculate_distance(
                    seed_attraction.lat_long[1], seed_attraction.lat_long[0],
                    attraction.lat_long[1], attraction.lat_long[0]
                )
                distances.append((attraction, dist))
        
        # Sort by distance (ascending)
        distances.sort(key=lambda x: x[1])
        
        # Remove seed and closest attractions
        destroyed.tourist.remove(seed_attraction)
        destroyed.unassigned.append(seed_attraction)
        
        # Remove closest attractions
        for i in range(min(num_to_remove-1, len(distances))):
            attraction = distances[i][0]
            destroyed.tourist.remove(attraction)
            destroyed.unassigned.append(attraction)
    
    return destroyed

def destroy_most_frequent_category(current, random_state):
    """
    Removes attractions from the most common category in the current solution.
    """
    destroyed = current.copy()

    category_counts = {}
    for day in destroyed.tourist.locations:
        for attr in destroyed.tourist.locations[day]:
            for cat in attr.categories:
                category_counts[cat] = category_counts.get(cat, 0) + 1

    if not category_counts:
        return destroyed

    # Identify most common category
    most_common_cat = max(category_counts, key=category_counts.get)

    # Get all attractions with that category
    to_remove = []
    for day in destroyed.tourist.locations:
        for attr in destroyed.tourist.locations[day]:
            if most_common_cat in attr.categories:
                to_remove.append(attr)

    destroy_factor=0.5
    num_to_remove = max(1, int(len(to_remove) * destroy_factor))
    selected = random_state.choice(to_remove, size=min(num_to_remove, len(to_remove)), replace=False)

    for attr in selected:
        destroyed.tourist.remove(attr)
        destroyed.unassigned.append(attr)

    return destroyed

def destroy_recently_added(current, random_state):
    """
    Removes the top 30% most recently added attractions (latest start times).
    Targets last location before going home
    """
    destroyed = current.copy()

    start_time_pairs = []
    for day in destroyed.tourist.start_times:
        for attr_name, start_time in destroyed.tourist.start_times[day].items():
            for attr in destroyed.tourist.locations[day]:
                if attr.attraction_name == attr_name:
                    start_time_pairs.append((day, attr, start_time))

    # Sort by latest start time
    start_time_pairs.sort(key=lambda x: x[2], reverse=True)

    destroy_factor=0.3
    num_to_remove = max(1, int(len(start_time_pairs) * destroy_factor))
    for i in range(min(num_to_remove, len(start_time_pairs))):
        day, attr, _ = start_time_pairs[i]
        destroyed.tourist.remove(attr)
        destroyed.unassigned.append(attr)

    return destroyed

def destroy_expensive(current, random_state):
    """
    Removes the top 50% most expensive attractions from the itinerary.
    """
    destroyed = current.copy()
    assigned_attractions = []
    for day in destroyed.tourist.locations:
        assigned_attractions.extend(destroyed.tourist.locations[day])

    if not assigned_attractions:
        return destroyed

    # Sort by cost descending
    assigned_attractions.sort(key=lambda a: a.cost, reverse=True)
    destroy_factor=0.5
    num_to_remove = max(1, int(len(assigned_attractions) * destroy_factor))
    to_remove = assigned_attractions[:num_to_remove]

    for attr in to_remove:
        destroyed.tourist.remove(attr)
        destroyed.unassigned.append(attr)

    return destroyed

### Repair operators ###
def repair_greedy(destroyed : SMJSP, random_state):
    """
    Repairs solution by greedily inserting attractions by fun score.
    
    Args:
        destroyed: SMJSP object after destroying
        random_state: numpy.random.RandomState
    
    Returns:
        repaired: SMJSP object after repairing
    """
    repaired = destroyed.copy()
    
    if not repaired.unassigned:
        return repaired
    
    # Make sure included_categories is up to date
    repaired.update_included_categories()
    
    # Sort attractions by fun score (descending)
    # This will include diversity bonus, encouraging new categories
    scored_attractions = []
    for attr in repaired.unassigned:
        # Calculate fun score with diversity bonus
        fun_score = repaired.get_attraction_fun_score(attr)
        scored_attractions.append((fun_score, random_state.random(), attr))
    
    scored_attractions.sort(reverse=True)
    
    # Try to insert each unassigned attraction
    for _, _, attraction in scored_attractions:
        # Try each day
        for day in range(repaired.tourist.days):
            time_slot = repaired.find_available_time_slot(attraction, day)
            
            if time_slot is not None and repaired.tourist.can_assign(attraction, time_slot, day):
                repaired.tourist.assign(attraction, day, time_slot)
                if attraction in repaired.unassigned:
                    repaired.unassigned.remove(attraction)
                # Update included categories after each insertion
                # to ensure diversity bonus is calculated correctly
                repaired.update_included_categories()
                break
    
    return repaired

def repair_rainy(destroyed : SMJSP, random_state):
    """
    Possible that future days might be rainy again, so we need to repair
    in such a way that rain adverse events happen first to quickly clear them
    """
    repaired = destroyed.copy()
    
    if not repaired.unassigned:
        return repaired
    
    # Make sure included_categories is up to date
    repaired.update_included_categories()

    # Then we will try to assign events that will not be affected by the rain fist
    non_rain_events = []
    rain_events = []
    for attraction in repaired.unassigned:
        if "Sporty" in attraction.categories or \
            "Nature" in attraction.categories or \
            "Outdoor" in attraction.categories:
            rain_events.append(attraction)
        else:
            non_rain_events.append(attraction)

    # Assign no rain first
    for attraction in non_rain_events:
        for day in range(repaired.tourist.days):
            time_slot = repaired.find_available_time_slot(attraction, day)
            
            if time_slot is not None and repaired.tourist.can_assign(attraction, time_slot, day):
                repaired.tourist.assign(attraction, day, time_slot)
                if attraction in repaired.unassigned:
                    repaired.unassigned.remove(attraction)

                repaired.update_included_categories()
                break

    # Then we assign rain
    for attraction in rain_events:
        for day in range(repaired.tourist.days):
            time_slot = repaired.find_available_time_slot(attraction, day)
            
            if time_slot is not None and repaired.tourist.can_assign(attraction, time_slot, day):
                repaired.tourist.assign(attraction, day, time_slot)
                if attraction in repaired.unassigned:
                    repaired.unassigned.remove(attraction)

                repaired.update_included_categories()
                break

    return repaired

def repair_regret(destroyed, random_state):
    """
    Repairs solution using regret-based insertion.
    Prioritizes attractions that would be more difficult to insert later.
    
    Args:
        destroyed: SMJSP object after destroying
        random_state: numpy.random.RandomState
    
    Returns:
        repaired: SMJSP object after repairing
    """
    repaired = destroyed.copy()
    
    # Make sure included_categories is up to date
    repaired.update_included_categories()
    
    while repaired.unassigned:
        max_regret = -1
        best_attraction = None
        best_day = None
        best_time = None
        
        # Calculate regret for each unassigned attraction
        for attraction in repaired.unassigned:
            # Find valid insertions across all days
            valid_insertions = []
            
            for day in range(repaired.tourist.days):
                time_slot = repaired.find_available_time_slot(attraction, day)
                
                if time_slot is not None and repaired.tourist.can_assign(attraction, time_slot, day):
                    valid_insertions.append((day, time_slot))
            
            if not valid_insertions:
                continue
                
            # Sort by fun score and time slot quality
            fun_score = repaired.get_attraction_fun_score(attraction)
            valid_insertions.sort(key=lambda x: x[1])  # Sort by earliest time slot
            
            # Calculate regret (difficulty of insertion)
            # More regret if few insertion options
            regret = fun_score * (1 + 1/(len(valid_insertions)))
            
            if regret > max_regret:
                max_regret = regret
                best_attraction = attraction
                best_day, best_time = valid_insertions[0]  # Choose earliest time slot
        
        if best_attraction is None:
            break  # No more feasible insertions
            
        # Insert the attraction with highest regret
        repaired.tourist.assign(best_attraction, best_day, best_time)
        repaired.unassigned.remove(best_attraction)
        
        # Update included categories after insertion
        repaired.update_included_categories()
    
    return repaired
def repair_random(destroyed, random_state):
    """
    Repairs solution by randomly inserting attractions where possible.
    
    Args:
        destroyed: SMJSP object after destroying
        random_state: numpy.random.RandomState
    
    Returns:
        repaired: SMJSP object after repairing
    """
    repaired = destroyed.copy()
    
    # Shuffle unassigned attractions
    unassigned_list = list(repaired.unassigned)
    random_state.shuffle(unassigned_list)
    
    # Try to insert each attraction
    for attraction in unassigned_list:
        # Try days in random order
        days = list(range(repaired.tourist.days))
        random_state.shuffle(days)
        
        for day in days:
            time_slot = repaired.find_available_time_slot(attraction, day)
            
            if time_slot is not None and repaired.tourist.can_assign(attraction, time_slot, day):
                repaired.tourist.assign(attraction, day, time_slot)
                if attraction in repaired.unassigned:
                    repaired.unassigned.remove(attraction)
                break
    
    return repaired

def repair_balanced(destroyed, random_state):
    """
    Repairs solution by trying to balance attractions across days.
    
    Args:
        destroyed: SMJSP object after destroying
        random_state: numpy.random.RandomState
    
    Returns:
        repaired: SMJSP object after repairing
    """
    repaired = destroyed.copy()
    
    if not repaired.unassigned:
        return repaired
    
    # Make sure included_categories is up to date
    repaired.update_included_categories()
    
    # Count attractions per day
    day_counts = [0] * repaired.tourist.days
    for day in range(repaired.tourist.days):
        if day in repaired.tourist.locations:
            day_counts[day] = len(repaired.tourist.locations[day])
    
    # Sort attractions by fun score
    scored_attractions = []
    for attr in repaired.unassigned:
        # Calculate fun score with diversity bonus
        fun_score = repaired.get_attraction_fun_score(attr)
        scored_attractions.append((fun_score, random_state.random(), attr))
    
    scored_attractions.sort(reverse=True)
    
    # Try to insert each unassigned attraction
    for _, _, attraction in scored_attractions:
        # Sort days by current number of attractions (ascending)
        days_sorted = sorted(range(repaired.tourist.days), key=lambda d: day_counts[d])
        
        # Try to insert into days with fewer attractions first
        for day in days_sorted:
            time_slot = repaired.find_available_time_slot(attraction, day)
            
            if time_slot is not None and repaired.tourist.can_assign(attraction, time_slot, day):
                repaired.tourist.assign(attraction, day, time_slot)
                if attraction in repaired.unassigned:
                    repaired.unassigned.remove(attraction)
                day_counts[day] += 1
                
                # Update included categories after insertion
                repaired.update_included_categories()
                break
    
    return repaired

def repair_nearest_neighbor(destroyed, random_state):
    """
    Repairs solution by inserting attractions close to existing ones.
    
    Args:
        destroyed: SMJSP object after destroying
        random_state: numpy.random.RandomState
    
    Returns:
        repaired: SMJSP object after repairing
    """
    repaired = destroyed.copy()
    
    if not repaired.unassigned:
        return repaired
    
    # Process day by day
    for day in range(repaired.tourist.days):
        if day not in repaired.tourist.locations:
            continue
            
        existing_locations = repaired.tourist.locations[day]
        
        # If day has attractions
        if existing_locations:
            # Keep trying to add attractions until no more can be added
            added_something = True
            while added_something and repaired.unassigned:
                added_something = False
                
                # Find nearest unassigned attraction to any existing attraction
                best_dist = float('inf')
                best_attr = None
                best_time = None
                
                for attraction in repaired.unassigned:
                    # Check if we can assign to this day
                    time_slot = repaired.find_available_time_slot(attraction, day)
                    
                    if time_slot is not None and repaired.tourist.can_assign(attraction, time_slot, day):
                        # Calculate minimum distance to any existing attraction
                        min_dist = float('inf')
                        for existing in existing_locations:
                            dist = calculate_distance(
                                attraction.lat_long[1], attraction.lat_long[0],
                                existing.lat_long[1], existing.lat_long[0]
                            )
                            min_dist = min(min_dist, dist)
                        
                        # Track the closest
                        if min_dist < best_dist:
                            best_dist = min_dist
                            best_attr = attraction
                            best_time = time_slot
                
                # Insert the closest attraction
                if best_attr is not None:
                    repaired.tourist.assign(best_attr, day, best_time)
                    repaired.unassigned.remove(best_attr)
                    existing_locations = repaired.tourist.locations[day]  # Update locations
                    added_something = True
    
    # Try greedy repair for any remaining unassigned attractions
    return repair_greedy(repaired, random_state)

def repair_maximize_attractions(destroyed, random_state):
    """
    Repairs solution by trying to fit as many attractions as possible.
    Prioritizes filling empty days.
    """
    repaired = destroyed.copy()
    
    # Get days with no attractions first
    empty_days = [day for day in range(repaired.tourist.days) 
                if day not in repaired.tourist.locations or not repaired.tourist.locations[day]]
    
    # Then fill other days
    all_days = empty_days + [day for day in range(repaired.tourist.days) if day not in empty_days]
    
    # Try to add attractions to each day
    for day in all_days:
        # Keep adding until we can't fit more or reach 3 attractions
        current_count = len(repaired.tourist.locations.get(day, []))
        while current_count < 3 and repaired.unassigned:
            # Find best attraction that fits
            best_attr = None
            best_time = None
            
            for attr in repaired.unassigned:
                time_slot = repaired.find_available_time_slot(attr, day)
                if time_slot is not None and repaired.tourist.can_assign(attr, time_slot, day):
                    if best_attr is None or repaired.get_attraction_fun_score(attr) > repaired.get_attraction_fun_score(best_attr):
                        best_attr = attr
                        best_time = time_slot
            
            if best_attr:
                repaired.tourist.assign(best_attr, day, best_time)
                repaired.unassigned.remove(best_attr)
                current_count += 1
                repaired.update_included_categories()
            else:
                break  # No more attractions can be assigned to this day
    
    return repaired

def repair_cheapest_first(destroyed, random_state):
    """
    Repairs solution by inserting cheapest attractions first.
    """
    repaired = destroyed.copy()
    if not repaired.unassigned:
        return repaired

    attractions = [(attr.cost, random_state.random(), attr) for attr in repaired.unassigned]
    attractions.sort()  # Ascending by cost

    for _, _, attraction in attractions:
        for day in range(repaired.tourist.days):
            time_slot = repaired.find_available_time_slot(attraction, day)
            if time_slot is not None and repaired.tourist.can_assign(attraction, time_slot, day):
                repaired.tourist.assign(attraction, day, time_slot)
                repaired.unassigned.remove(attraction)
                repaired.update_included_categories()
                break

    return repaired

def repair_diversity_first(destroyed, random_state):
    """
    Repairs solution by prioritizing attractions from unseen categories.
    """
    repaired = destroyed.copy()
    repaired.update_included_categories()

    # Score based on number of unseen categories
    scored = []
    for attr in repaired.unassigned:
        new_cats = [c for c in attr.categories if c not in repaired.included_categories]
        score = len(new_cats)
        scored.append((score, random_state.random(), attr))

    scored.sort(reverse=True)

    for _, _, attraction in scored:
        for day in range(repaired.tourist.days):
            time_slot = repaired.find_available_time_slot(attraction, day)
            if time_slot is not None and repaired.tourist.can_assign(attraction, time_slot, day):
                repaired.tourist.assign(attraction, day, time_slot)
                repaired.unassigned.remove(attraction)
                repaired.update_included_categories()
                break

    return repaired

def repair_time_slot_fit(destroyed, random_state):
    """
    Repairs solution by inserting attractions that are easiest to schedule.
    """
    repaired = destroyed.copy()

    attraction_flex = []
    for attr in repaired.unassigned:
        feasible_days = 0
        for day in range(repaired.tourist.days):
            time_slot = repaired.find_available_time_slot(attr, day)
            if time_slot is not None and repaired.tourist.can_assign(attr, time_slot, day):
                feasible_days += 1
        if feasible_days > 0:
            attraction_flex.append((feasible_days, random_state.random(), attr))

    attraction_flex.sort(reverse=True)

    for _, _, attraction in attraction_flex:
        for day in range(repaired.tourist.days):
            time_slot = repaired.find_available_time_slot(attraction, day)
            if time_slot is not None and repaired.tourist.can_assign(attraction, time_slot, day):
                repaired.tourist.assign(attraction, day, time_slot)
                repaired.unassigned.remove(attraction)
                repaired.update_included_categories()
                break

    return repaired

def repair_cheapest_fun_proximity(destroyed, random_state):
    """
    Repairs by prioritizing cheapest attractions, then fun score, then proximity to hotel.
    """
    repaired = destroyed.copy()
    hotel_lat, hotel_long = repaired.tourist.hotel_lat, repaired.tourist.hotel_long

    scored = []
    for attr in repaired.unassigned:
        fun_score = repaired.get_attraction_fun_score(attr)
        dist = calculate_distance(hotel_lat, hotel_long, attr.lat_long[1], attr.lat_long[0])
        score = (attr.cost, -fun_score, dist)  # Sort by cost (asc), fun (desc), dist (asc)
        scored.append((score, random_state.random(), attr))

    scored.sort()

    for _, _, attraction in scored:
        for day in range(repaired.tourist.days):
            time_slot = repaired.find_available_time_slot(attraction, day)
            if time_slot is not None and repaired.tourist.can_assign(attraction, time_slot, day):
                repaired.tourist.assign(attraction, day, time_slot)
                repaired.unassigned.remove(attraction)
                repaired.update_included_categories()
                break

    return repaired

from disruptions import (
    rainy_day_attraction, heat_wave_attraction, early_closure,
    nothing_happens_attraction, family_day
)
from utils import calculate_distance

def is_affected_by_disruption(attraction, day, disruption_type):
    """
    Check if an attraction would be affected by a specific disruption type.
    
    Args:
        attraction: Attraction object
        day: Day index
        disruption_type: Type of disruption (rainy, heat_wave, etc.)
        
    Returns:
        bool: True if attraction would be affected
    """
    # Check if the attraction is open on this day
    if day not in attraction.opening_hours:
        return True  # Already not available
    
    # Check disruption-specific constraints
    if disruption_type == "rainy" and "Outdoor" in attraction.categories:
        return True
    
    elif disruption_type == "heat_wave" and any(
            cat in attraction.categories for cat in ["Sporty", "Nature", "Outdoor"]):
        return True
    
    elif disruption_type == "early_closure" and "Outdoor" not in attraction.categories:
        # Check if attraction would extend past noon
        opening_time = attraction.opening_hours[day][0]
        if opening_time >= 12 or (opening_time + attraction.task_time) > 12:
            return True
    
    elif disruption_type == "family_day" and "Family" not in attraction.categories:
        return True
    
    return False

def repair_minimize_changes(destroyed, random_state):
    """
    Repairs solution by minimizing changes to the original itinerary.
    Prioritizes replacement attractions near original locations.
    
    Args:
        destroyed: SMJSP object after disruptions
        random_state: numpy.random.RandomState
        
    Returns:
        repaired: SMJSP object with minimized changes
    """
    repaired = destroyed.copy()
    
    # Check if original plan is available
    if not hasattr(repaired, 'original_plan') or not repaired.original_plan:
        # Fall back to standard repair if no original plan
        return repair_greedy(repaired, random_state)
    
    # Get disruption forecasts if available
    disruption_forecasts = {}
    if hasattr(repaired, 'scenario_info') and 'forecasts' in repaired.scenario_info:
        disruption_forecasts = repaired.scenario_info['forecasts']
    
    # For each day with disruptions
    for day in range(repaired.tourist.days):
        # Skip if day has max attractions already
        if day in repaired.tourist.locations and len(repaired.tourist.locations[day]) >= 3:
            continue
            
        # Get disruption type for this day
        disruption_type = disruption_forecasts.get(str(day), "normal")
        
        # Compare what was originally planned vs what remains
        if day in repaired.original_plan:
            original_attrs = repaired.original_plan[day]
            current_attrs = repaired.tourist.locations.get(day, [])
            
            # Find attractions that were removed due to disruption
            # This is the problematic part - original_attrs may not have the expected format
            removed_attrs = []
            for item in original_attrs:
                # Check if it's a tuple or an Attraction object
                if isinstance(item, tuple) and len(item) >= 2:
                    attr, start_time = item[0], item[1]
                    # Check if this attraction is not in current attractions
                    if not any(curr.attraction_name == attr.attraction_name for curr in current_attrs):
                        removed_attrs.append((attr, start_time))
                elif hasattr(item, 'attraction_name'):  # It's just an Attraction object
                    attr = item
                    # Try to find a start time if available
                    start_time = repaired.tourist.start_times.get(day, {}).get(attr.attraction_name, 9.0)
                    if not any(curr.attraction_name == attr.attraction_name for curr in current_attrs):
                        removed_attrs.append((attr, start_time))
            
            # For each removed attraction, try to find a nearby replacement
            for attr, start_time in removed_attrs:
                if day in repaired.tourist.locations and len(repaired.tourist.locations[day]) >= 3:
                    break  # Day already has max attractions
                
                location = (attr.lat_long[1], attr.lat_long[0])
                
                # Find unassigned attractions that are feasible and close to the original
                candidates = []
                for unassigned in repaired.unassigned:
                    # Skip attractions affected by the same disruption
                    if is_affected_by_disruption(unassigned, day, disruption_type):
                        continue
                        
                    # Check if can be assigned at similar time
                    time_slot = repaired.find_available_time_slot(unassigned, day)
                    if time_slot is None or not repaired.tourist.can_assign(unassigned, time_slot, day):
                        continue
                    
                    # Calculate distance from original attraction
                    unassigned_loc = (unassigned.lat_long[1], unassigned.lat_long[0])
                    distance = calculate_distance(
                        location[0], location[1], 
                        unassigned_loc[0], unassigned_loc[1]
                    )
                    
                    # Prefer start times close to original
                    time_diff = abs(time_slot - start_time)
                    
                    # Assign a score combining distance and schedule similarity
                    similarity_score = distance * 0.7 + time_diff * 0.3
                    candidates.append((unassigned, time_slot, similarity_score))
                
                # Sort by similarity score (lower is better)
                candidates.sort(key=lambda x: x[2])
                
                # Assign the best candidate if any exist
                if candidates:
                    best_attr, best_time, _ = candidates[0]
                    repaired.tourist.assign(best_attr, day, best_time)
                    repaired.unassigned.remove(best_attr)
    
    # Use standard repair method to fill in any remaining capacity
    if repaired.unassigned:
        repaired = repair_greedy(repaired, random_state)
    
    return repaired