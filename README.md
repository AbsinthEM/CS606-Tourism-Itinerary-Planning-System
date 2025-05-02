# CS606_GroupAssignment

This the ALNS group project

# Tourism Itinerary Planning System

This project implements an intelligent tourism itinerary planning system for Singapore attractions using the Adaptive Large Neighborhood Search (ALNS) metaheuristic. The system creates personalized itineraries based on tourist preferences, budget constraints, and available time while optimizing for enjoyment, variety, and travel efficiency.

## Problem Definition

The tourism itinerary planning problem involves scheduling a set of attractions for a tourist over multiple days. Each tourist has specific preferences (e.g., Nature, Cultural, Shopping), a budget, and a fixed number of days to spend in Singapore. Each attraction is characterized by:

- Geographic location (latitude/longitude)
- Visit duration
- Opening hours (which can vary by day)
- Cost
- Categories (e.g., Cultural, Nature, Family-friendly)
- Other attributes

The goal is to create an optimal itinerary that maximizes tourist satisfaction while respecting all constraints.

## Constraints

The system handles multiple real-world constraints:

1. **Budget Constraints**: Total attraction costs cannot exceed the tourist's budget
2. **Time Constraints**:
   - Attractions can only be visited during their opening hours
   - Tourist visits are limited to their daily touring hours (e.g., 10 AM - 6 PM)
   - Each attraction has a specific visit duration
3. **Capacity Constraints**: Maximum 3 attractions per day
4. **Break Time**: Minimum 1 hour total break time between attractions per day
5. **Travel Constraints**: Travel time between attractions based on geographic distances
6. **No Overlapping**: A tourist cannot be at two places simultaneously

## Objective Function

The system optimizes a multi-objective function that considers three components with respective weights that are normalized to sum to 100%:

1. **Fun Score** (default weight: 0.6, ~46% after normalization):

   - Match between tourist preferences and attraction categories
   - Preference matching follows exponential scaling (multiple matches are disproportionately rewarded)
   - Diversity bonus for including different categories of attractions

2. **Travel Efficiency** (default weight: 0.4, ~31% after normalization):

   - Minimizing total travel distance
   - Considering hotel location as start/end point each day

3. **Number of Attractions** (default weight: 0.3, ~23% after normalization):
   - Maximizing the number of attractions visited
   - Penalizing empty days

The implementation automatically normalizes these weights to ensure they sum to 1.0. The objective function is then negated for use with the ALNS minimization framework.

## Key Design Decisions

### ALNS Framework

The implementation uses the Adaptive Large Neighborhood Search metaheuristic, which iteratively destroys and repairs solutions to explore the search space effectively. The ALNS framework:

- Maintains a set of destroy and repair operators
- Uses adaptive weights to select operators based on past performance
- Employs simulated annealing for accepting/rejecting candidate solutions

### Diversity Promotion

The system promotes diversity in itineraries through:

- A category diversity bonus that rewards including new types of attractions
- Balancing attractions across days
- Prioritizing preference matches while still allowing variety

### Dynamic Replanning

The system can handle disruptions like weather changes:

- Rainy day scenario: Outdoor attractions become unavailable
- Sick day scenario: All attractions become unavailable for that day
- Replanning algorithm adjusts itineraries in response to disruptions

## Implementation Components

### Data Structures

- `Tourist`: Represents the tourist with preferences, budget, touring hours, and schedule
- `Attraction`: Represents attractions with location, opening hours, categories, and visit duration
- `SMJSP`: Single Machine Job Scheduling Problem, the core problem formulation class

### ALNS Operators

#### Destroy Operators

1. **Random Destroy**: Randomly removes a percentage of attractions
2. **Worst Destroy**: Removes attractions with lowest contribution to objective
3. **Day Destroy**: Removes all attractions from a random day
4. **Preference Destroy**: Removes attractions least matching tourist preferences
5. **Cluster Destroy**: Removes geographically clustered attractions

#### Repair Operators

1. **Greedy Repair**: Inserts attractions by descending fun score
2. **Regret Repair**: Prioritizes attractions that would be difficult to insert later
3. **Random Repair**: Randomly inserts attractions where possible
4. **Balanced Repair**: Balances attractions across days
5. **Nearest Neighbor Repair**: Inserts attractions close to existing ones
6. **Maximize Attractions Repair**: Fits as many attractions as possible

### Solution Evaluation

- Fun score calculation with preference matching and diversity bonuses
- Travel distance calculation using Haversine formula
- Day-by-day itinerary generation with start/end times
- Objective normalization for comparable component weights

## Usage

To run ALNS:
```python
python alns_main.py <attraction_data_csv> <tourist_data_csv> <random_seed> [options]
```

To run RL Training:
```python
python dr_alns_trainer.py
```

To run RL with ALNS:
```python
python dr_alns_main.py
```


### Optional Arguments:

- `--tourist_id`: Specific tourist ID to plan for
- `--diversity_bonus`: Bonus points for first attraction of each category
- `--fun_weight`: Weight for fun score in objective function (default: 0.6)
- `--distance_weight`: Weight for distance in objective function (default: 0.4)
- `--attraction_weight`: Weight for number of attractions in objective function (default: 0.3)

## Evaluation

The system evaluates itinerary quality through:

1. **Objective Improvement**: Tracking objective value improvement from initial to final solution
2. **Dynamic Replanning Evaluation**: Testing robustness under different weather scenarios
3. **Operator Performance**: Analyzing which destroy/repair operators perform best
4. **Preference Satisfaction**: Measuring how well the itinerary matches tourist preferences
5. **Budget Utilization**: Optimizing enjoyment within budget constraints

## Results Output

The system generates detailed outputs:

- Day-by-day itinerary with attractions, times, and costs
- Travel distances and times between locations
- Categories covered in the itinerary
- Total budget spent
- Objective score breakdown

## Future Improvements

- Group tourism planning with shared preferences
- Multi-day attraction visits
- Time-dependent travel estimation
- Integration with real-time data (weather, crowd levels)
- User feedback incorporation

## Dependencies

- Python 3.6+
- NumPy
- Matplotlib
- tqdm (for progress bars)

# React App Guide and Requirements

1. Install the following: https://nodejs.org/en
2. Install the dependencies: npm install
3. Open 1 terminal to start the backend:
     - cd to backend and run the following command: py server.py
4. Open another terminal to start the frontend:
     - cd to dynamic-itinerary-planner folder and run the following command: npm start

<<<<<<< HEAD
=======

>>>>>>> cb4171e6908bfd501bb4f08e6abd7dc0ff6250a8
