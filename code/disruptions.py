"""
This module handles possible disruptions to a day's itinerary.
"""

from rcjsp import Attraction, Tourist
from typing import List
import copy

def rainy_day_attraction(attraction_list: List[Attraction], day: int) -> List[Attraction]:
    """
    Rainy Day:
    Any attraction categorized as Outdoor only
    becomes unavailable (closes) for the given 'day'.
    """
    # print("Rainy Day")
    new_attraction_list = []

    
    for attraction in attraction_list:
        # If it's an outdoor-related attraction, remove opening hours for that day
        if ("Outdoor" in attraction.categories):
            if day in attraction.opening_hours:
                del attraction.opening_hours[day]
        new_attraction_list.append(attraction)
    return new_attraction_list


def early_closure(attraction_list: List[Attraction], day: int) -> List[Attraction]:
    """
    Early Closure:
    All NOT 'Outdoor' attractions close at midday (12:00) for the given 'day'.
    """
    # print("Early Closure")
    new_attraction_list = []
    
    for attraction in attraction_list:
        # Create a deep copy to avoid modifying original objects
        new_attr = copy.deepcopy(attraction)
        
        # If it's not an outdoor attraction, modify its closing time
        if "Outdoor" not in new_attr.categories:
            if day in new_attr.opening_hours:
                # The format is usually [open_time, close_time]
                new_attr.opening_hours[day][1] = min(12, new_attr.opening_hours[day][1])
        
        # Add ALL attractions to the new list (both outdoor and non-outdoor)
        new_attraction_list.append(new_attr)
        
    return new_attraction_list


def nothing_happens_attraction(attraction_list: List[Attraction], day: int) -> List[Attraction]:
    """
    Nothing Happens:
    Everything is normal, no changes to opening hours.
    """
    # print("Normal Day")
    return attraction_list

def heat_wave_attraction(attraction_list: List[Attraction], day: int) -> List[Attraction]:
    """
    Heat Wave:
    If an attraction is Sporty, Nature, or Outdoor,
    it becomes unavailable (closes) for the given 'day'.
    """

    # print("Heat Wave Day")
    new_attraction_list = []
    for attraction in attraction_list:
        # Check categories, remove opening hours for that day
        if ("Sporty" in attraction.categories or 
            "Nature" in attraction.categories or 
            "Outdoor" in attraction.categories):
            if day in attraction.opening_hours:
                del attraction.opening_hours[day]
        new_attraction_list.append(attraction)
    

    return new_attraction_list


def family_day(attraction_list: List[Attraction], day: int) -> List[Attraction]:
    """
    We can only visit family attractions today
    """
    new_attraction_list = []

        
    for attraction in attraction_list:
        # Check the family category, remove day from opening_hours
        if ("Family" not in attraction.categories):
            if day in attraction.opening_hours:
                del attraction.opening_hours[day]
        new_attraction_list.append(attraction)

    return new_attraction_list
