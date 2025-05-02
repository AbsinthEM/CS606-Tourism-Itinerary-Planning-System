# Common evaluation module

def calculate_objective_metrics(solution):
    """
    Calculate standard objective metrics from a solution.
    Returns a dictionary of metrics.
    """
    obj_negative = solution.objective()
    obj_positive = -obj_negative
    
    # Get detailed breakdown
    _, breakdown = solution.objective(return_breakdown=True)
    
    # Count attractions
    total_attractions = 0
    for day in solution.tourist.locations:
        total_attractions += len(solution.tourist.locations[day])
    
    return {
        "objective_negative": obj_negative,
        "objective_positive": obj_positive,
        "fun_score": breakdown["fun_score_raw"],
        "distance": breakdown["distance_raw"],
        "attractions_count": total_attractions,
        "money_spent": solution.tourist.money_spent,
        "normalized_fun": breakdown["normalized_fun"],
        "normalized_distance": breakdown["normalized_distance"],
        "normalized_attractions": breakdown["normalized_attractions"],
        "breakdown": breakdown
    }

def compare_solutions(solution1_metrics, solution2_metrics, solution1_name="Initial", solution2_name="Optimized"):
    """
    Compare two solutions and produce standardized comparison metrics.
    Returns a dictionary with comparison results.
    """
    obj1 = solution1_metrics["objective_positive"]
    obj2 = solution2_metrics["objective_positive"]
    
    # Calculate component changes if detailed metrics are available
    component_changes = {}
    for component in ["normalized_fun", "normalized_distance", "normalized_attractions"]:
        if component in solution1_metrics and component in solution2_metrics:
            val1 = solution1_metrics[component]
            val2 = solution2_metrics[component]
            if val1 > 0:
                pct_change = ((val2 / val1) - 1) * 100
            else:
                pct_change = float('inf') if val2 > 0 else 0
            component_changes[component] = pct_change
    
    # Calculate attraction change if available
    if "attractions_count" in solution1_metrics and "attractions_count" in solution2_metrics:
        attr_count1 = solution1_metrics["attractions_count"]
        attr_count2 = solution2_metrics["attractions_count"]
        attr_change = attr_count2 - attr_count1
        if attr_count1 > 0:
            attr_change_pct = ((attr_count2 / attr_count1) - 1) * 100
        else:
            attr_change_pct = float('inf') if attr_count2 > 0 else 0
    else:
        attr_change = 0
        attr_change_pct = 0
    
    # Case 1: Both positive
    if obj1 >= 0 and obj2 >= 0:
        if obj1 > 0:  # Avoid division by zero
            improvement_pct = ((obj2 - obj1) / obj1) * 100
        else:
            improvement_pct = float('inf') if obj2 > 0 else 0
        description = f"{improvement_pct:.1f}%"
        
    # Case 2: Solution1 negative, Solution2 positive
    elif obj1 < 0 and obj2 >= 0:
        gain = abs(obj1) + obj2
        description = f"Complete reversal from {obj1:.4f} to {obj2:.4f} (positive gain of {gain:.4f})"
        improvement_pct = 100  # Set to 100% for sorting/comparison
        
    # Case 3: Solution1 positive, Solution2 negative
    elif obj1 >= 0 and obj2 < 0:
        loss = abs(obj2) + obj1
        description = f"Deterioration from {obj1:.4f} to {obj2:.4f} (loss of {loss:.4f})"
        improvement_pct = -100  # Set to -100% for sorting/comparison
        
    # Case 4: Both negative
    else:
        if obj1 != 0:
            penalty_reduction = (abs(obj1) - abs(obj2)) / abs(obj1) * 100
        else:
            penalty_reduction = float('inf') if obj2 > obj1 else float('-inf')
        description = f"Penalty reduction of {penalty_reduction:.1f}% ({obj1:.4f} to {obj2:.4f})"
        improvement_pct = penalty_reduction
    
    # Determine which solution is better
    better_solution = solution2_name if obj2 > obj1 else solution1_name
    
    
    return {
        "improvement_description": description,
        "improvement_percentage": improvement_pct,
        "better_solution": better_solution,
        "component_changes": component_changes,
        "attraction_change": attr_change,
        "attraction_change_pct": attr_change_pct
    }