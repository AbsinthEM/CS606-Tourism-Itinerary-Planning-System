"""
Feasibility checker for Stage 1 tourism itinerary planning.
Validates if a plan (saved in JSON format) meets all project basic constraints.
"""

import os
import json
import argparse
import csv
from rcjsp import Attraction
from utils import calculate_distance


class FeasibilityCheckerStage1:
    """
    Class to check the feasibility of Stage 1 tourism itinerary plans.
    """
    def __init__(self, plan_json_path, attractions_csv_path):
        """
        Initialize with paths to plan JSON and attraction data CSV.

        Args:
            plan_json_path: Path to the solution JSON file
            attractions_csv_path: Path to the attractions data CSV file
        """
        self.plan_json_path = plan_json_path
        self.attractions_csv_path = attractions_csv_path
        self.plan_data = None
        self.attractions_data = {}

        # Load data
        self._load_plan_data()
        self._load_attractions_data()

        # Validation results
        self.validation_results = {
            "is_valid": True,
            "violations": [],
            "warnings": [],
            "total_cost": 0,
            "total_attractions": 0,
            "days_with_attractions": 0
        }

    def _load_plan_data(self):
        """Load the plan data from JSON file."""
        try:
            with open(self.plan_json_path, 'r') as f:
                self.plan_data = json.load(f)

            # Normalize preferences (remove any single quotes)
            if "tourist" in self.plan_data and "preferences" in self.plan_data["tourist"]:
                clean_preferences = []
                for pref in self.plan_data["tourist"]["preferences"]:
                    # Strip any surrounding single quotes
                    cleaned = pref.strip("'")
                    clean_preferences.append(cleaned)

                self.plan_data["tourist"]["preferences"] = clean_preferences

            print(f"Successfully loaded plan data from {self.plan_json_path}")
        except Exception as e:
            print(f"Error loading plan data: {e}")
            raise

    def _load_attractions_data(self):
        """Load and process attractions data from CSV file."""
        try:
            with open(self.attractions_csv_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header row

                for idx, row in enumerate(reader):
                    if not row:
                        continue

                    # Use the original Attraction class from your project
                    attraction = Attraction(row, idx)

                    # Adjust opening hours to use 0-indexed days
                    adjusted_opening_hours = {}
                    for day, hours in attraction.opening_hours.items():
                        adjusted_opening_hours[day - 1] = hours
                    attraction.opening_hours = adjusted_opening_hours

                    self.attractions_data[attraction.attraction_name] = attraction

            print(f"Successfully loaded {len(self.attractions_data)} attractions from {self.attractions_csv_path}")
        except Exception as e:
            print(f"Error loading attractions data: {e}")
            raise

    def validate(self):
        """
        Validate the plan against all Stage 1 constraints.
        Returns a dictionary with validation results.
        """
        # Track validation status
        self.validation_results["plan_id"] = os.path.basename(self.plan_json_path)

        # Perform validation checks
        self._validate_budget_constraint()
        self._validate_capacity_constraint()
        self._validate_time_constraints()
        self._validate_break_time_constraint()
        self._validate_travel_constraints()

        # Calculate summary stats
        self._calculate_summary_stats()

        # Print summary
        self._print_validation_summary()

        return self.validation_results

    def _validate_budget_constraint(self):
        """Validate that the total cost doesn't exceed the tourist's budget."""
        tourist = self.plan_data["tourist"]
        budget = tourist["budget"]
        total_cost = 0

        for day, attractions in tourist["locations"].items():
            for attraction_name in attractions:
                if attraction_name in self.attractions_data:
                    total_cost += self.attractions_data[attraction_name].cost

        self.validation_results["total_cost"] = total_cost

        if total_cost > budget:
            self.validation_results["is_valid"] = False
            self.validation_results["violations"].append(
                f"Budget exceeded: Spent ${total_cost} but budget is ${budget}"
            )

    def _validate_capacity_constraint(self):
        """Validate that each day has a maximum of 3 attractions."""
        tourist = self.plan_data["tourist"]

        for day, attractions in tourist["locations"].items():
            if len(attractions) > 3:
                self.validation_results["is_valid"] = False
                self.validation_results["violations"].append(
                    f"Capacity constraint violated: Day {int(day)+1} has {len(attractions)} attractions (max allowed: 3)"
                )

    def _validate_time_constraints(self):
        """
        Validate time-related constraints:
        - Attractions visited during opening hours
        - Visits within tourist's touring hours
        - No overlapping visits
        """
        tourist = self.plan_data["tourist"]
        touring_hours = tourist["touring_hours"]

        for day_str, attractions in tourist["locations"].items():
            day = int(day_str)
            day_start_times = tourist["start_times"].get(day_str, {})

            if not day_start_times:
                continue

            # Sort attractions by start time
            sorted_attractions = []
            for attraction_name in attractions:
                if attraction_name in day_start_times and attraction_name in self.attractions_data:
                    start_time = day_start_times[attraction_name]
                    attr = self.attractions_data[attraction_name]
                    end_time = start_time + attr.task_time
                    sorted_attractions.append((attraction_name, start_time, end_time, attr))

            sorted_attractions.sort(key=lambda x: x[1])

            # Check for overlapping visits
            for i in range(len(sorted_attractions) - 1):
                curr_name, curr_start, curr_end, curr_attr = sorted_attractions[i]
                next_name, next_start, next_end, next_attr = sorted_attractions[i+1]

                if curr_end > next_start:
                    self.validation_results["is_valid"] = False
                    self.validation_results["violations"].append(
                        f"Overlapping visits on Day {day+1}: {curr_name} ({curr_start:.1f}-{curr_end:.1f}) and "
                        f"{next_name} ({next_start:.1f}-{next_end:.1f})"
                    )

            # Check for attractions' opening hours and tourist's touring hours
            for name, start_time, end_time, attr in sorted_attractions:
                # Tourist's touring hours
                if start_time < touring_hours[0] or end_time > touring_hours[1]:
                    self.validation_results["is_valid"] = False
                    self.validation_results["violations"].append(
                        f"Visit outside touring hours on Day {day+1}: {name} ({start_time:.1f}-{end_time:.1f}) "
                        f"but touring hours are {touring_hours[0]}-{touring_hours[1]}"
                    )

                # Attraction's opening hours
                if day in attr.opening_hours:
                    opening_time, closing_time = attr.opening_hours[day]
                    if start_time < opening_time or end_time > closing_time:
                        self.validation_results["is_valid"] = False
                        self.validation_results["violations"].append(
                            f"Visit outside opening hours on Day {day+1}: {name} ({start_time:.1f}-{end_time:.1f}) "
                            f"but opening hours are {opening_time}-{closing_time}"
                        )
                else:
                    self.validation_results["is_valid"] = False
                    self.validation_results["violations"].append(
                        f"Attraction closed on Day {day+1}: {name} has no opening hours for this day"
                    )

    def _validate_break_time_constraint(self):
        """Validate that there's at least 1 hour of total break time between attractions per day."""
        tourist = self.plan_data["tourist"]

        for day_str, attractions in tourist["locations"].items():
            day = int(day_str)
            day_start_times = tourist["start_times"].get(day_str, {})

            if not day_start_times or len(attractions) <= 1:
                continue

            # Sort attractions by start time
            time_slots = []
            for attraction_name in attractions:
                if attraction_name in day_start_times and attraction_name in self.attractions_data:
                    start_time = day_start_times[attraction_name]
                    end_time = start_time + self.attractions_data[attraction_name].task_time
                    time_slots.append((start_time, end_time))

            time_slots.sort()

            # Calculate total break time
            total_break_time = 0
            for i in range(len(time_slots) - 1):
                curr_end = time_slots[i][1]
                next_start = time_slots[i+1][0]
                break_time = next_start - curr_end
                total_break_time += max(0, break_time)  # Ensure no negative break time

            if total_break_time < 1 and len(time_slots) > 1:
                self.validation_results["is_valid"] = False
                self.validation_results["violations"].append(
                    f"Insufficient break time on Day {day+1}: Only {total_break_time:.1f} hours (minimum required: 1 hour)"
                )

    def _validate_travel_constraints(self):
        """Validate that there's sufficient time for travel between attractions."""
        tourist = self.plan_data["tourist"]
        hotel_lat = tourist["hotel_lat"]
        hotel_long = tourist["hotel_long"]

        for day_str, attractions in tourist["locations"].items():
            day = int(day_str)
            day_start_times = tourist["start_times"].get(day_str, {})

            if not day_start_times:
                continue

            # Sort attractions by start time
            sorted_attractions = []
            for attraction_name in attractions:
                if attraction_name in day_start_times and attraction_name in self.attractions_data:
                    start_time = day_start_times[attraction_name]
                    attr = self.attractions_data[attraction_name]
                    end_time = start_time + attr.task_time
                    sorted_attractions.append((attraction_name, start_time, end_time, attr))

            sorted_attractions.sort(key=lambda x: x[1])

            # Check travel time from hotel to first attraction
            if sorted_attractions:
                first_name, first_start, _, first_attr = sorted_attractions[0]
                first_lat, first_long = first_attr.lat_long[1], first_attr.lat_long[0]
                distance = calculate_distance(hotel_lat, hotel_long, first_lat, first_long)
                travel_time = distance / 30.0  # 30 km/h average speed

                if first_start < tourist["touring_hours"][0] + travel_time:
                    self.validation_results["is_valid"] = False
                    self.validation_results["violations"].append(
                        f"Insufficient travel time on Day {day+1}: Cannot reach {first_name} by {first_start:.1f} "
                        f"from hotel (requires {travel_time:.1f} hours)"
                    )

            # Check travel time between attractions
            for i in range(len(sorted_attractions) - 1):
                curr_name, curr_start, curr_end, curr_attr = sorted_attractions[i]
                next_name, next_start, next_end, next_attr = sorted_attractions[i+1]

                curr_lat, curr_long = curr_attr.lat_long[1], curr_attr.lat_long[0]
                next_lat, next_long = next_attr.lat_long[1], next_attr.lat_long[0]

                distance = calculate_distance(curr_lat, curr_long, next_lat, next_long)
                travel_time = distance / 30.0  # 30 km/h average speed

                if next_start < curr_end + travel_time:
                    self.validation_results["is_valid"] = False
                    self.validation_results["violations"].append(
                        f"Insufficient travel time on Day {day+1}: Cannot reach {next_name} by {next_start:.1f} "
                        f"from {curr_name} (requires {travel_time:.1f} hours)"
                    )

            # Check travel time from last attraction back to hotel
            if sorted_attractions:
                last_name, last_start, last_end, last_attr = sorted_attractions[-1]
                last_lat, last_long = last_attr.lat_long[1], last_attr.lat_long[0]
                distance = calculate_distance(last_lat, last_long, hotel_lat, hotel_long)
                travel_time = distance / 30.0  # 30 km/h average speed

                if last_end + travel_time > tourist["touring_hours"][1]:
                    self.validation_results["is_valid"] = False
                    self.validation_results["violations"].append(
                        f"Insufficient travel time on Day {day+1}: Cannot return to hotel by {tourist['touring_hours'][1]} "
                        f"from {last_name} (requires {travel_time:.1f} hours)"
                    )

    def _calculate_summary_stats(self):
        """Calculate summary statistics for the plan."""
        tourist = self.plan_data["tourist"]
        total_attractions = 0
        days_with_attractions = 0

        for day_str, attractions in tourist["locations"].items():
            if attractions:
                total_attractions += len(attractions)
                days_with_attractions += 1

        self.validation_results["total_attractions"] = total_attractions
        self.validation_results["days_with_attractions"] = days_with_attractions

        # Count missing categories from preferences
        preferences = tourist["preferences"]
        covered_categories = set()

        for day_str, attractions in tourist["locations"].items():
            for attraction_name in attractions:
                if attraction_name in self.attractions_data:
                    for category in self.attractions_data[attraction_name].categories:
                        covered_categories.add(category)

        missing_preferences = [pref for pref in preferences if pref not in covered_categories]
        if missing_preferences:
            self.validation_results["warnings"].append(
                f"Preferences not covered: {', '.join(missing_preferences)}"
            )

    def _print_validation_summary(self):
        """Print a summary of the validation results."""
        print("\n" + "="*50)
        print(f"VALIDATION SUMMARY FOR: {os.path.basename(self.plan_json_path)}")
        print("="*50)

        if self.validation_results["is_valid"]:
            print("✅ VALID: Plan meets all constraints!")
        else:
            print("❌ INVALID: Plan violates one or more constraints!")

        print(f"\nTotal attractions: {self.validation_results['total_attractions']}")
        print(f"Days with attractions: {self.validation_results['days_with_attractions']}/{self.plan_data['tourist']['days']}")
        print(f"Total cost: ${self.validation_results['total_cost']}/{self.plan_data['tourist']['budget']} (Budget)")

        if self.validation_results["violations"]:
            print("\nVIOLATIONS:")
            for i, violation in enumerate(self.validation_results["violations"]):
                print(f"{i+1}. {violation}")

        if self.validation_results["warnings"]:
            print("\nWARNINGS:")
            for i, warning in enumerate(self.validation_results["warnings"]):
                print(f"{i+1}. {warning}")

        print("\n" + "="*50)


def main():
    """Main function to run the validation."""
    parser = argparse.ArgumentParser(description="Check feasibility of Stage 1 tourism itinerary plans")
    parser.add_argument("plan_json", help="Path to the Stage 1 solution JSON file")
    parser.add_argument("attractions_csv", help="Path to the attractions data CSV file")
    parser.add_argument("--output", help="Path to save validation results (optional)")

    args = parser.parse_args()

    # Create validator and run validation
    validator = FeasibilityCheckerStage1(args.plan_json, args.attractions_csv)
    results = validator.validate()

    # Save results if output path provided
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nValidation results saved to: {args.output}")


if __name__ == "__main__":
    main()