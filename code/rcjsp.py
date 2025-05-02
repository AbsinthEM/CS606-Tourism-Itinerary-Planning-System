import copy
import json
import random
import csv
import ast
from src.alns import State
from typing import List
import math
from datetime import datetime
from utils import calculate_distance


class Attraction(object):
    def __init__(self, attraction_data : list, idx : int):
        self.type_list = ["Cultural",
                        "Sporty",
                        "Nature",
                        "Family",
                        "Shopping",
                        "Culinary",
                        "Outdoor"]

        self.idx = idx
        self.attraction_name = attraction_data[0]
        self.lat_long = [float(attraction_data[1]), float(attraction_data[2])]
        self.task_time = float(attraction_data[3])
        self.opening_hours = ast.literal_eval(attraction_data[4])

        # Adjust opening hours to check for formatting
        for key in self.opening_hours.keys():
            hours = self.opening_hours[key]
            new_hours = []
            if int(hours[0]) != float(hours[0]):
                # Half hours interval
                new_hours.append(float(int(hours[0])) + 0.5)
            else:
                new_hours.append(float(hours[0]))
            
            if int(hours[1]) != float(hours[1]):
                # Half hours interval
                new_hours.append(float(int(hours[1])) + 0.5)
            else:
                new_hours.append(float(hours[1]))

            self.opening_hours[key] = new_hours

        self.cost = int(attraction_data[5])
        self.location_encoding = attraction_data[6:]

        self.categories = []
        # Each location has more than one category
        for index, encoding in enumerate(self.location_encoding):
            if int(encoding) == 1:
                self.categories.append(self.type_list[index])

        # print(self.opening_hours)


class Tourist(object):
    def __init__(self, tourist_data : list):
        self.idx = tourist_data[0]
        self.preferences = tourist_data[1].strip("[]").split(",")
        self.budget = int(tourist_data[2])
        # self.must_visit = ast.literal_eval(tourist_data[3]) # remove for now for simplicity
        self.days = int(tourist_data[3])
        self.touring_hours = ast.literal_eval(tourist_data[4])
        self.hotel_long = float(tourist_data[5]) # added starting location as hotel
        self.hotel_lat = float(tourist_data[6])  # added starting location as hotel
        

        # Create lists and dict to hold items
        self.money_spent = 0

        # Available free time
        self.touring_dict = {}
        for i in range(self.days):
            self.touring_dict[i] = self.touring_hours

        # Format is key is day, then inside is list of Attraction Class
        # e.g. {1: ["Marina Bay", "Lau Par Sat"], 2: ["Lakeside Park"]}
        # names used for convention, but we keep track of tasks for better lookup
        self.locations = {}

        # Format is key is day, then within the visiting there is 
        # another dicitonary which contains the locations and the start time of visit
        # e.g. {1: {"Marina Bay": 8.5, "Lau Par Sat": 13}, 2: {"Lakeside Park": 7}} etc.
        self.start_times = {}

        # This list contain the list of visited locations 
        self.visited = []

    
    def can_assign(self, attraction : Attraction, time : int, day : int) -> bool:
        """
        Check whether you can assign the attraction to the tourist when he/she visits
        at a given time at a given day, accounting for travel time between attractions
        """
        from utils import calculate_distance  # Import the distance calculation function
        
        # If the attraction is already assigned to another day, return False. Need to check all days
        for d in self.locations.keys():
            if attraction in self.locations[d]:
                return False
            
        # Cannot exceed Budget
        if attraction.cost + self.money_spent > self.budget:
            return False
        
        # End time cannot be later than end of touring hours or 
        # Start time cannot be earlier as start of touring hours
        if time < self.touring_dict[day][0]:
            return False
        if time + attraction.task_time > self.touring_dict[day][1]:
            return False
        
        # Check if locations dictionary for this day has been initialized
        if day in self.locations and len(self.locations[day]) >= 3:
            return False  # 3 activities per day max

        # Visitors must visit during visiting hours  
        if day not in attraction.opening_hours.keys():
            return False
        else:
            cur_opening_hours = attraction.opening_hours[day]
            # too early
            if time < cur_opening_hours[0]:
                return False
            
            # too late
            if time + attraction.task_time > cur_opening_hours[1]:
                return False

        # Initialize if needed
        if day not in self.locations:
            self.locations[day] = []
        if day not in self.start_times:
            self.start_times[day] = {}
        
        # Get existing attractions with their start and end times
        time_slots = []
        for existing_attr in self.locations[day]:
            if existing_attr.attraction_name in self.start_times[day]:
                start_time = self.start_times[day][existing_attr.attraction_name]
                end_time = start_time + existing_attr.task_time
                time_slots.append((existing_attr, start_time, end_time))
        
        # Sort by start time
        time_slots.sort(key=lambda x: x[1])
        
        # Calculate end time of new attraction
        new_end_time = time + attraction.task_time
        
        # Find where the new attraction would fit in the schedule
        insertion_index = 0
        while insertion_index < len(time_slots) and time_slots[insertion_index][1] < time:
            insertion_index += 1
        
        # Check travel time from previous attraction/hotel to this attraction
        if insertion_index == 0:
            # First attraction of the day - travel from hotel
            prev_lat = self.hotel_lat
            prev_long = self.hotel_long
            prev_end_time = self.touring_dict[day][0]  # Day start time
        else:
            # Travel from previous attraction
            prev_attr, _, prev_end_time = time_slots[insertion_index - 1]
            prev_lat = prev_attr.lat_long[1]
            prev_long = prev_attr.lat_long[0]
        
        # Calculate travel time from previous location
        distance_from_prev = calculate_distance(
            prev_lat, prev_long, 
            attraction.lat_long[1], attraction.lat_long[0]
        )
        travel_time_from_prev = distance_from_prev / 30.0  # 30 km/h average speed
        
        # Check if there's enough time to travel from previous location
        if prev_end_time + travel_time_from_prev > time:
            return False
        
        # Check travel time to next attraction/hotel
        if insertion_index < len(time_slots):
            # Travel to next attraction
            next_attr, next_start_time, _ = time_slots[insertion_index]
            next_lat = next_attr.lat_long[1]
            next_long = next_attr.lat_long[0]
            
            # Calculate travel time to next location
            distance_to_next = calculate_distance(
                attraction.lat_long[1], attraction.lat_long[0],
                next_lat, next_long
            )
            travel_time_to_next = distance_to_next / 30.0  # 30 km/h average speed
            
            # Check if there's enough time to travel to next location
            if new_end_time + travel_time_to_next > next_start_time:
                return False
        else:
            # Last attraction - check time to return to hotel
            hotel_lat = self.hotel_lat
            hotel_long = self.hotel_long
            
            # Calculate travel time back to hotel
            distance_to_hotel = calculate_distance(
                attraction.lat_long[1], attraction.lat_long[0],
                hotel_lat, hotel_long
            )
            travel_time_to_hotel = distance_to_hotel / 30.0  # 30 km/h average speed
            
            # Check if there's enough time to return to hotel before day ends
            if new_end_time + travel_time_to_hotel > self.touring_dict[day][1]:
                return False
        
        # Check for overlaps in time slots
        all_time_slots = [(start, end) for _, start, end in time_slots]
        all_time_slots.append((time, new_end_time))
        all_time_slots.sort()
        
        for i in range(len(all_time_slots) - 1):
            if all_time_slots[i][1] > all_time_slots[i+1][0]:
                return False
        
        # Calculate total break time in the day
        total_break_time = 0
        for i in range(len(all_time_slots) - 1):
            break_time = all_time_slots[i+1][0] - all_time_slots[i][1]
            total_break_time += break_time
        
        # Ensure at least 1 hour total break time per day
        # Skip this check if there's only one activity
        if len(all_time_slots) > 1 and total_break_time < 1:
            return False

        return True
    
    def assign(self, attraction : Attraction, day : int, time : float) -> None:
        """
        Assign the attraction for the specific day and time
        """
        # Add attraction to the locations
        if day not in self.locations.keys():
            self.locations[day] = [attraction]
        else:
            self.locations[day].append(attraction)

        # Add the start time
        if day not in self.start_times:
            self.start_times[day] = {attraction.attraction_name : time}
        else:
            self.start_times[day][attraction.attraction_name] = time

        self.money_spent = 0
        for day in self.locations.keys():
            for attr in self.locations[day]:
                self.money_spent += attr.cost


    def remove(self, attraction : Attraction) -> None:
        """
        Remove attraction from dictionaries
        """
        for key in self.locations.keys():
            if attraction in self.locations[key]:
                self.locations[key].remove(attraction)

        # Remove the start time from the start times
        for key in self.locations.keys():
            if attraction.attraction_name in self.start_times[key].keys():
                del self.start_times[key][attraction.attraction_name]
        

### Parser to parse instance json file ###
class Parser(object):
    def __init__(self, attraction_csv, tourist_csv):
        """
        Parse all information, turn into usable infomration
        """
        self.attraction_csv = attraction_csv
        self.tourist_csv    = tourist_csv

        self.attraction_data = []
        self.tourist_data = []

        with open(self.attraction_csv, "r") as af:
            attr_reader = csv.reader(af)

            next(attr_reader)

            for row in attr_reader:
                self.attraction_data.append(row)

        with open(self.tourist_csv, "r") as tf:
            tour_reader = csv.reader(tf)

            next(tour_reader)

            for row in tour_reader:
                self.tourist_data.append(row)
        
        # We then need to parse the data into useable data
        self.attractions = [Attraction(data, idx) for idx, data in enumerate(self.attraction_data)]
        self.tourists = [Tourist(data) for data in self.tourist_data]

class SMJSP(State):
    def __init__(self, 
                tourist : Tourist, 
                attractions : List[Attraction],
                weighting : list = [0.6, 0.4],
                diversity_bonus: int = 15):  # Added diversity bonus parameter
        """
        Single Machine Job Scheduling Problem

        Tourist : Tourist Class to plan for
        Attractions : List of available Attraction
        Weighting : The weighting between [fun score, travel distance]
        diversity_bonus : Additional fun score points for first attraction of each category
        """
        self.tourist = tourist
        self.attractions = attractions
        self.weighting = weighting
        self.diversity_bonus = diversity_bonus
        
        # Initialize empty solution
        self.unassigned = list(attractions)
        
        # Initialize empty dictionaries for each day
        for i in range(self.tourist.days):
            if i not in self.tourist.locations:
                self.tourist.locations[i] = []
            if i not in self.tourist.start_times:
                self.tourist.start_times[i] = {}
        
        # Initialize an empty set to track categories that have been included in the itinerary
        self.included_categories = set()
        
    def get_attraction_fun_score(self, attraction, include_diversity_bonus=True):
        """
        Calculate fun score of an attraction based on tourist preferences
        with optional diversity bonus for first attraction of each category.
        
        Modified to strongly prioritize preference matches.
        """
        # Base fun score
        base_fun_score = 30
        
        # Count how many preference matches instead of binary matching
        preference_matches = 0
        for category in attraction.categories:
            if category in self.tourist.preferences:
                preference_matches += 1
        
        # Calculate preference multiplier (0 if no matches, 1.0-4.0 if matches)
        if preference_matches > 0:
            # Exponential scaling to strongly prefer matches
            preference_multiplier = 1.0 + (preference_matches * 1.5)
        else:
            preference_multiplier = 0.2  # Small non-zero value for non-matching attractions
        
        # Calculate basic fun score with stronger scaling
        fun_score = base_fun_score * preference_multiplier
        
        # Add diversity bonus for new categories if requested
        if include_diversity_bonus and preference_multiplier > 0.5:
            new_categories = [cat for cat in attraction.categories if cat not in self.included_categories]
            if new_categories:
                # Scale diversity bonus by preference multiplier
                # This ensures diversity within preferred categories is prioritized
                diversity_value = self.diversity_bonus * len(new_categories) * preference_multiplier / 4.0
                fun_score += diversity_value
        
        return fun_score
    
    def update_included_categories(self):
        """
        Update the set of categories that have been included in the current itinerary.
        This should be called whenever the solution changes.
        """
        self.included_categories = set()
        for day in range(self.tourist.days):
            if day in self.tourist.locations:
                for attraction in self.tourist.locations[day]:
                    for category in attraction.categories:
                        self.included_categories.add(category)


    def calculate_travel_time(self, from_lat, from_long, to_lat, to_long):
        """
        Calculate travel time between two locations.
        Assumes average speed of 30 km/h in Singapore
        Returns travel time in hours
        """
        distance = calculate_distance(from_lat, from_long, to_lat, to_long)
        # Assume average speed of 30 km/h for transportation in Singapore
        return distance / 30.0
        
    def calculate_max_distance(self):
        """
        Calculate a more accurate maximum possible travel distance based on
        actual coordinates of attractions and hotel rather than a fixed estimate.
        """
        # Find max possible distance between any two points
        max_single_distance = 0
        
        # Add hotel coordinates to the mix
        all_coords = [(self.tourist.hotel_lat, self.tourist.hotel_long)]
        
        # Add all attraction coordinates
        for attr in self.attractions:
            all_coords.append((attr.lat_long[1], attr.lat_long[0]))
        
        # Find maximum distance between any two points
        for i in range(len(all_coords)):
            for j in range(i+1, len(all_coords)):
                lat1, lon1 = all_coords[i]
                lat2, lon2 = all_coords[j]
                dist = calculate_distance(lat1, lon1, lat2, lon2)
                max_single_distance = max(max_single_distance, dist)
        
        # Maximum trips per day:
        # - Up to 3 attractions per day
        # - Going from hotel to first attraction
        # - Going between attractions (up to 2 transitions)
        # - Returning from last attraction to hotel
        max_trips_per_day = 4  # hotel -> A1 -> A2 -> A3 -> hotel
        
        # Estimate maximum total distance across all days
        return max_single_distance * max_trips_per_day * self.tourist.days

    
    def find_available_time_slot(self, attraction, day):
        """
        Find the earliest available time slot for an attraction on a given day,
        accounting for travel time between attractions
        """
        from utils import calculate_distance
        
        # Get touring hours for the day
        day_start, day_end = self.tourist.touring_dict[day]
        
        # Get attraction opening hours for the day
        if day not in attraction.opening_hours:
            return None
        
        attraction_open, attraction_close = attraction.opening_hours[day]
        
        # Duration of the visit
        duration = attraction.task_time
        
        # If no activities planned for this day yet, we can start at the earliest time
        # after adding travel time from hotel
        if day not in self.tourist.locations or not self.tourist.locations[day]:
            # Calculate travel time from hotel
            hotel_lat = self.tourist.hotel_lat
            hotel_long = self.tourist.hotel_long
            
            distance = calculate_distance(
                hotel_lat, hotel_long,
                attraction.lat_long[1], attraction.lat_long[0]
            )
            travel_time = distance / 30.0  # 30 km/h
            
            # Earliest possible start after traveling from hotel
            earliest_start = max(day_start + travel_time, attraction_open)
            
            # Check if we can complete the visit before closing/day end
            if earliest_start + duration <= min(day_end, attraction_close):
                return earliest_start
            return None
        
        # Get all scheduled activities for the day
        activities = self.tourist.locations[day]
        start_times = self.tourist.start_times[day]
        
        # Convert to list of (attraction, start_time, end_time) and sort
        time_slots = []
        for act in activities:
            if act.attraction_name in start_times:
                start = start_times[act.attraction_name]
                end = start + act.task_time
                time_slots.append((act, start, end))
        
        time_slots.sort(key=lambda x: x[1])  # Sort by start time
        
        # Try to find a suitable time slot
        
        # 1. Check before first attraction
        if time_slots:
            first_attr, first_start, _ = time_slots[0]
            
            # Travel time from hotel to new attraction
            hotel_to_new = calculate_distance(
                self.tourist.hotel_lat, self.tourist.hotel_long,
                attraction.lat_long[1], attraction.lat_long[0]
            ) / 30.0
            
            # Travel time from new attraction to first attraction
            new_to_first = calculate_distance(
                attraction.lat_long[1], attraction.lat_long[0],
                first_attr.lat_long[1], first_attr.lat_long[0]
            ) / 30.0
            
            earliest_start = max(day_start + hotel_to_new, attraction_open)
            latest_end = first_start - new_to_first
            
            if earliest_start + duration <= min(latest_end, attraction_close):
                return earliest_start
        
        # 2. Check between existing attractions
        for i in range(len(time_slots) - 1):
            curr_attr, _, curr_end = time_slots[i]
            next_attr, next_start, _ = time_slots[i + 1]
            
            # Travel time from current to new attraction
            curr_to_new = calculate_distance(
                curr_attr.lat_long[1], curr_attr.lat_long[0],
                attraction.lat_long[1], attraction.lat_long[0]
            ) / 30.0
            
            # Travel time from new to next attraction
            new_to_next = calculate_distance(
                attraction.lat_long[1], attraction.lat_long[0],
                next_attr.lat_long[1], next_attr.lat_long[0]
            ) / 30.0
            
            earliest_start = max(curr_end + curr_to_new, attraction_open)
            latest_end = next_start - new_to_next
            
            if earliest_start + duration <= min(latest_end, attraction_close):
                return earliest_start
        
        # 3. Check after last attraction
        if time_slots:
            last_attr, _, last_end = time_slots[-1]
            
            # Travel time from last to new attraction
            last_to_new = calculate_distance(
                last_attr.lat_long[1], last_attr.lat_long[0],
                attraction.lat_long[1], attraction.lat_long[0]
            ) / 30.0
            
            # Travel time from new attraction to hotel
            new_to_hotel = calculate_distance(
                attraction.lat_long[1], attraction.lat_long[0],
                self.tourist.hotel_lat, self.tourist.hotel_long
            ) / 30.0
            
            earliest_start = max(last_end + last_to_new, attraction_open)
            latest_end = day_end - new_to_hotel
            
            if earliest_start + duration <= min(latest_end, attraction_close):
                return earliest_start
        
        # No suitable time slot found
        return None
        
    def random_initialize(self, seed=None):
        """
        Create an initial solution using a greedy construction heuristic
        
        Args:
            seed::int
                random seed
        Returns:
            objective::float
                objective value of the state
        """
        if seed is None:
            seed = 606

        random.seed(seed)
        
        # Sort attractions by fun score (descending)
        scored_attractions = [(self.get_attraction_fun_score(attr), random.random(), attr) 
                            for attr in self.attractions]
        # Sort by score (descending), then by random value for tie-breaking
        scored_attractions.sort(reverse=True)
        
        # Try to assign each attraction
        for _, _, attraction in scored_attractions:
            # Try each day to find a suitable time slot
            for day in range(self.tourist.days):
                time_slot = self.find_available_time_slot(attraction, day)
                
                if time_slot is not None and self.tourist.can_assign(attraction, time_slot, day):
                    self.tourist.assign(attraction, day, time_slot)
                    if attraction in self.unassigned:
                        self.unassigned.remove(attraction)
                    break
        
        return self.objective()
        
    def copy(self):
        return copy.deepcopy(self)

    def objective(self, return_breakdown=False):
        """
        Calculate the objective value of the state.
        Higher is better conceptually (maximization problem),
        but return value is negated to work with ALNS's minimization approach.
        Balances fun satisfaction, travel efficiency, and maximizing number of sites.
        """
        # Fun satisfaction: sum of fun scores for assigned attractions
        fun_score = 0
        # Travel efficiency: negative of total travel distance
        total_distance = 0
        # Count of total attractions
        total_attractions = 0
        
        # Collect all assigned attractions first
        assigned_attractions = []
        for day in range(self.tourist.days):
            if day in self.tourist.locations:
                assigned_attractions.extend(self.tourist.locations[day])

        # Process each day: start at the hotel each day
        for day in range(self.tourist.days):
            if day not in self.tourist.locations:
                continue

            prev_lat = self.tourist.hotel_lat
            prev_long = self.tourist.hotel_long

            activities = self.tourist.locations[day]
            if not activities:
                continue

            # Count attractions for this day
            total_attractions += len(activities)

            # Filter activities with valid start times
            activities_with_times = []
            for act in activities:
                if act.attraction_name in self.tourist.start_times.get(day, {}):
                    activities_with_times.append((act, self.tourist.start_times[day][act.attraction_name]))

            if not activities_with_times:
                continue

            activities_with_times.sort(key=lambda x: x[1])
            for attraction, _ in activities_with_times:
                fun_score += self.get_attraction_fun_score(attraction)
                curr_lat, curr_long = attraction.lat_long[1], attraction.lat_long[0]
                distance = calculate_distance(prev_lat, prev_long, curr_lat, curr_long)
                total_distance += distance
                prev_lat, prev_long = curr_lat, curr_long

            # Add distance to return to hotel
            total_distance += calculate_distance(prev_lat, prev_long, self.tourist.hotel_lat, self.tourist.hotel_long)

        # Calculate average number of categories per attraction
        avg_categories = 1  # Default fallback
        if len(self.attractions) > 0:
            avg_categories = sum(len(attr.categories) for attr in self.attractions) / len(self.attractions)

        
        # Improved normalize components - use fixed maximum based on maximum possible attractions
        max_fun_score = self.tourist.days * 3 * (30*1.5 + self.diversity_bonus * avg_categories) # use fixed max of 3 instead of current for better weight normalization
        max_distance = self.calculate_max_distance()            # dynamic maximum distance
        max_attractions = self.tourist.days * 3                 # maximum attractions (3 per day)

        # Enhanced: only penalize excessive travel distance
        reasonable_distance_per_attraction = 10  # km
        expected_distance = total_attractions * reasonable_distance_per_attraction
        excess_distance = max(0, total_distance - expected_distance)
        
        norm_fun = fun_score / max_fun_score if max_fun_score > 0 else 0
        norm_distance = 1 - (excess_distance / max_distance if max_distance > 0 else 0)  # use excess distance instead of total distance
        norm_attractions = total_attractions / max_attractions if max_attractions > 0 else 0
        
    

        # Calculate empty day penalty (empty days are penalized)
        empty_days = sum(1 for day in range(self.tourist.days)
                        if day not in self.tourist.locations or not self.tourist.locations[day])
        empty_day_penalty = empty_days * 5
    

        # Weights: assuming self.weighting = [fun_weight, distance_weight, attraction_weight]
        fun_weight = self.weighting[0] if len(self.weighting) >= 1 else 0.4
        distance_weight = self.weighting[1] if len(self.weighting) >= 2 else 0.2
        attraction_weight = self.weighting[2] if len(self.weighting) >= 3 else 0.4

        # Calculate the weighted objective value
        objective_value = (
            fun_weight * norm_fun +
            distance_weight * norm_distance +
            attraction_weight * norm_attractions -
            empty_day_penalty
        )
        
    
        # Prepare breakdown dictionary
        breakdown = {
            "fun_score_raw": fun_score,
            "distance_raw": total_distance,
            "total_attractions": total_attractions,
            "empty_days": empty_days,
            "normalized_fun": norm_fun,
            "normalized_distance": norm_distance,
            "normalized_attractions": norm_attractions,
            "weighted_fun_contribution": fun_weight * norm_fun,
            "weighted_distance_contribution": distance_weight * norm_distance,
            "weighted_attractions_contribution": attraction_weight * norm_attractions,
            "weighted_objective_positive": objective_value,
            "weighted_objective_negative": -objective_value
        }
        
        if return_breakdown:
            return -objective_value, breakdown
        else:
            return -objective_value

    def save_to_file(self, filename):
        """Save SMJSP solution to a file."""
        solution_data = {
            'tourist': {
                'idx': self.tourist.idx,
                'preferences': self.tourist.preferences,
                'budget': self.tourist.budget,
                'days': self.tourist.days,
                'touring_hours': self.tourist.touring_hours,
                'hotel_long': self.tourist.hotel_long,
                'hotel_lat': self.tourist.hotel_lat,
                'money_spent': self.tourist.money_spent,
                'locations': {},
                'start_times': {}
            },
            'included_categories': list(self.included_categories),
            'unassigned': []
        }

        # Save locations and start times
        for day in self.tourist.locations:
            # Convert attraction objects to attraction names (for serialization)
            solution_data['tourist']['locations'][day] = [attr.attraction_name for attr in self.tourist.locations[day]]
            solution_data['tourist']['start_times'][day] = self.tourist.start_times[day]

        # Save unassigned attractions (by name)
        solution_data['unassigned'] = [attr.attraction_name for attr in self.unassigned]

        # Save to file
        with open(filename, 'w') as f:
            json.dump(solution_data, f, indent=2)
        
    def get_slack_ratio(self):
        """
        Calculate the average slack ratio across all scheduled activities.
        Slack = time between the end of an attraction and the next one
        Returns a normalized slack ratio per day (0 = tight, high = more slack)
        """
        total_slack = 0
        slack_counts = 0

        for day in range(self.tourist.days):
            if day not in self.tourist.locations:
                continue

            activities = self.tourist.locations[day]
            if len(activities) < 2:
                continue

            # Get start times
            times = self.tourist.start_times.get(day, {})
            sorted_activities = sorted(
                [(a, times[a.attraction_name]) for a in activities if a.attraction_name in times],
                key=lambda x: x[1]
            )

            for i in range(len(sorted_activities) - 1):
                end_time_current = sorted_activities[i][1] + sorted_activities[i][0].task_time
                start_time_next = sorted_activities[i + 1][1]
                slack = max(0, start_time_next - end_time_current)
                total_slack += slack
                slack_counts += 1

        return total_slack / slack_counts if slack_counts > 0 else 0.0

    def calculate_solution_diversity(self, prev_solutions):
        """
        Compute a simple diversity score based on Jaccard similarity of attraction names.
        Compares current assigned attractions to previous ones.
        """
        current_set = set()
        for day in self.tourist.locations:
            current_set.update([attr.attraction_name for attr in self.tourist.locations[day]])

        if not prev_solutions:
            return 1.0  # Max diversity if no history

        diversity_scores = []
        for prev in prev_solutions:
            prev_set = set()
            for d in prev.tourist.locations:
                prev_set.update([a.attraction_name for a in prev.tourist.locations[d]])
            intersection = len(current_set & prev_set)
            union = len(current_set | prev_set)
            if union == 0:
                diversity_scores.append(1.0)
            else:
                diversity_scores.append(1.0 - (intersection / union))  # Dissimilarity

        return sum(diversity_scores) / len(diversity_scores)

    @classmethod
    def load_from_file(cls, filename, attractions):
        """
        Load SMJSP solution from a file.
        
        Args:
            filename: Path to the saved solution file
            attractions: List of Attraction objects (needed to reconstruct the solution)
            
        Returns:
            SMJSP object with the loaded solution
        """
        # Create attraction lookup by name
        attraction_lookup = {attr.attraction_name: attr for attr in attractions}
        
        # Load solution data
        with open(filename, 'r') as f:
            solution_data = json.load(f)
        
        # Recreate tourist
        tourist_data = solution_data['tourist']
        tourist = Tourist([
            tourist_data['idx'],
            str(tourist_data['preferences']),
            tourist_data['budget'],
            tourist_data['days'],
            str(tourist_data['touring_hours']),
            tourist_data['hotel_long'],
            tourist_data['hotel_lat']
        ])
        
        # Set properties that aren't set by the constructor
        tourist.money_spent = tourist_data['money_spent']
        
        # Create SMJSP instance
        smjsp = cls(tourist, attractions)
        
        # Restore included categories
        smjsp.included_categories = set(solution_data['included_categories'])
        
        # Restore locations and start times
        for day, attraction_names in tourist_data['locations'].items():
            day = int(day)  # JSON keys are strings
            smjsp.tourist.locations[day] = []
            for name in attraction_names:
                if name in attraction_lookup:
                    smjsp.tourist.locations[day].append(attraction_lookup[name])
        
        for day, times in tourist_data['start_times'].items():
            day = int(day)  # JSON keys are strings
            smjsp.tourist.start_times[day] = times
        
        # Restore unassigned attractions
        smjsp.unassigned = []
        for name in solution_data['unassigned']:
            if name in attraction_lookup:
                smjsp.unassigned.append(attraction_lookup[name])
        
        return smjsp
        
def get_objective_breakdown(smjsp):
    """Get detailed breakdown of objective components."""
    _, breakdown = smjsp.objective(return_breakdown=True)
    return breakdown

"""
Enhanced Stage 2 objective function implementation to incorporate:
1. Stability (preserving original itinerary attractions)
2. Proximity (choosing replacements close to original attractions)
"""

class EnhancedObjectiveSMJSP(SMJSP):
    """
    Enhanced SMJSP class with stability and proximity objective components.
    Inherits from the standard SMJSP class but adds enhanced objective function.
    """
    def __init__(self, 
                tourist,
                attractions,
                original_solution=None,
                weighting=[0.6, 0.4],
                diversity_bonus=15,
                stability_weight=0.3,
                proximity_weight=0.2):
        """
        Initialize with original SMJSP parameters plus:
        - original_solution: The pre-disruption solution to compare against
        - stability_weight: Weight for the stability component (0-1)
        - proximity_weight: Weight for the proximity component (0-1)
        """
        # Initialize using parent class constructor
        super().__init__(tourist, attractions, weighting, diversity_bonus)
        
        self.original_solution = original_solution
        self.stability_weight = stability_weight
        self.proximity_weight = proximity_weight
        
        # Store original itinerary information for quick lookup
        self.original_attractions = {}
        self.original_times = {}
        self.original_locations = {}
        
        if original_solution:
            # Store original attraction information by day
            for day in range(original_solution.tourist.days):
                if day in original_solution.tourist.locations:
                    self.original_attractions[day] = {}
                    self.original_times[day] = {}
                    self.original_locations[day] = []
                    
                    # Store attractions with their start times
                    for attr in original_solution.tourist.locations[day]:
                        if attr.attraction_name in original_solution.tourist.start_times[day]:
                            start_time = original_solution.tourist.start_times[day][attr.attraction_name]
                            self.original_attractions[day][attr.attraction_name] = attr
                            self.original_times[day][attr.attraction_name] = start_time
                            self.original_locations[day].append((attr, start_time))
    
    def calculate_stability_score(self):
        """
        Measure how much of the original itinerary is preserved.
        Returns a value between 0 (completely different) and 1 (identical).
        """
        if not self.original_solution:
            return 0.0
            
        total_original_attractions = 0
        preserved_attractions = 0
        
        for day in range(self.tourist.days):
            if day in self.original_attractions:
                original_attr_names = set(self.original_attractions[day].keys())
                total_original_attractions += len(original_attr_names)
                
                if day in self.tourist.locations:
                    current_attr_names = {attr.attraction_name for attr in self.tourist.locations[day]}
                    preserved_attractions += len(original_attr_names.intersection(current_attr_names))
        
        # Calculate stability as percentage of preserved attractions
        if total_original_attractions == 0:
            return 0.0
            
        return preserved_attractions / total_original_attractions
    
    def calculate_proximity_score(self):
        """
        For attractions that were replaced, calculate how close the replacements are.
        Returns a value between 0 (very far) and 1 (very close).
        """
        if not self.original_solution:
            return 0.0
            
        total_distance = 0.0
        replacement_count = 0
        
        # For each day
        for day in range(self.tourist.days):
            if day not in self.original_locations or day not in self.tourist.locations:
                continue
                
            # Skip days with no attractions in either solution
            if not self.original_locations[day] or not self.tourist.locations[day]:
                continue
            
            # Get current attractions with start times
            current_attractions = []
            for attr in self.tourist.locations[day]:
                if attr.attraction_name in self.tourist.start_times[day]:
                    start_time = self.tourist.start_times[day][attr.attraction_name]
                    current_attractions.append((attr, start_time))
            
            # Sort both sets by start time
            original_sorted = sorted(self.original_locations[day], key=lambda x: x[1])
            current_sorted = sorted(current_attractions, key=lambda x: x[1])
            
            # Match by position in schedule (first with first, second with second, etc.)
            for i in range(min(len(original_sorted), len(current_sorted))):
                orig_attr, _ = original_sorted[i]
                new_attr, _ = current_sorted[i]
                
                # If different attractions, calculate distance
                if orig_attr.attraction_name != new_attr.attraction_name:
                    distance = calculate_distance(
                        orig_attr.lat_long[1], orig_attr.lat_long[0],
                        new_attr.lat_long[1], new_attr.lat_long[0]
                    )
                    total_distance += distance
                    replacement_count += 1
        
        # Calculate average distance of replacements
        if replacement_count == 0:
            # No replacements needed, perfect proximity
            return 1.0
            
        avg_distance = total_distance / replacement_count
        
        # Convert to a score (lower distance = higher score)
        # Assume 15km is the maximum reasonable distance for a replacement
        max_distance = 15.0
        proximity_score = 1.0 - min(1.0, avg_distance / max_distance)
        
        return proximity_score
    
    def calculate_schedule_similarity(self):
        """
        Calculate how similar the timing of activities is between original and current solution.
        Returns a value between 0 (completely different times) and 1 (same times).
        """
        if not self.original_solution:
            return 0.0
            
        time_differences = []
        
        # For each day
        for day in range(self.tourist.days):
            if day not in self.original_attractions or day not in self.tourist.locations:
                continue
                
            # Check attractions that exist in both solutions
            for attr in self.tourist.locations[day]:
                # If this attraction was in the original solution on this day
                if attr.attraction_name in self.original_attractions[day]:
                    # Get the start times
                    if attr.attraction_name in self.tourist.start_times[day]:
                        current_start = self.tourist.start_times[day][attr.attraction_name]
                        original_start = self.original_times[day][attr.attraction_name]
                        
                        # Calculate time difference (in hours)
                        time_diff = abs(current_start - original_start)
                        time_differences.append(time_diff)
        
        # If no common attractions found
        if not time_differences:
            return 0.0
            
        # Calculate average time difference
        avg_diff = sum(time_differences) / len(time_differences)
        
        # Convert to similarity score (0-1)
        # Assume 3 hours difference is maximum (would give 0 score)
        max_diff = 3.0
        similarity = 1.0 - min(1.0, avg_diff / max_diff)
        
        return similarity
    
    def objective(self, return_breakdown=False):
        """
        Enhanced objective function that incorporates stability and proximity.
        """
        # Get the base objective calculation from parent class
        base_obj_negative, breakdown = super().objective(return_breakdown=True)
        base_obj_positive = -base_obj_negative
        
        # Only calculate enhanced components if original solution exists
        if self.original_solution:
            # Calculate stability score
            stability = self.calculate_stability_score()
            
            # Calculate proximity score for replacements
            proximity = self.calculate_proximity_score()
            
            # Calculate schedule similarity
            schedule_similarity = self.calculate_schedule_similarity()
            
            # Combine scores with weights
            enhanced_component = (
                (self.stability_weight * stability) + 
                (self.proximity_weight * proximity) + 
                (0.1 * schedule_similarity)  # Small weight for schedule similarity
            )
            
            # Combine with base objective (adjust weights to maintain scale)
            base_weight = 1.0 - self.stability_weight - self.proximity_weight - 0.1
            enhanced_obj_positive = (base_weight * base_obj_positive) + enhanced_component
            enhanced_obj_negative = -enhanced_obj_positive
            
            # Add enhanced components to breakdown
            breakdown["stability_score"] = stability
            breakdown["proximity_score"] = proximity
            breakdown["schedule_similarity"] = schedule_similarity
            breakdown["enhanced_component"] = enhanced_component
            breakdown["enhanced_obj_positive"] = enhanced_obj_positive
            breakdown["enhanced_obj_negative"] = enhanced_obj_negative
            
            if return_breakdown:
                return enhanced_obj_negative, breakdown
            else:
                return enhanced_obj_negative
        else:
            # No original solution to compare against, use base objective
            if return_breakdown:
                return base_obj_negative, breakdown
            else:
                return base_obj_negative


def create_enhanced_smjsp(original_solution, tourist, attractions, stability_weight=0.3, proximity_weight=0.2):
    """
    Factory function to create an EnhancedObjectiveSMJSP instance from an original solution.
    This preserves the original weighting and diversity bonus settings.
    
    Args:
        original_solution: The pre-disruption SMJSP solution
        tourist: Tourist object (typically from the disrupted solution)
        attractions: List of attractions
        stability_weight: Weight for stability component (0-1)
        proximity_weight: Weight for proximity component (0-1)
        
    Returns:
        An EnhancedObjectiveSMJSP instance
    """
    # Create enhanced SMJSP with same settings as original
    enhanced = EnhancedObjectiveSMJSP(
        tourist,
        attractions,
        original_solution=original_solution,
        weighting=original_solution.weighting,
        diversity_bonus=original_solution.diversity_bonus,
        stability_weight=stability_weight,
        proximity_weight=proximity_weight
    )
    
    # Copy other properties
    enhanced.unassigned = list(enhanced.unassigned)  # Create a new list, not a reference
    enhanced.included_categories = set(enhanced.included_categories)  # Create a new set, not a reference
    
    return enhanced