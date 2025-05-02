// src/App.js
import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import MapView from './components/MapView';
import SidePanel from './components/SidePanel';

function App() {
  //----------------------------------------------------------
  // 1) STATE: We hold the current `solution` + user inputs
  //----------------------------------------------------------
  const [solution, setSolution] = useState(null);

  // Basic ALNS user parameters
  const [budget, setBudget] = useState(1000);
  const [days, setDays] = useState(3);
  const [touringHours, setTouringHours] = useState([8, 19]);
  const [hotelLat, setHotelLat] = useState(1.2868);
  const [hotelLong, setHotelLong] = useState(103.8545);
  const [preferences, setPreferences] = useState(["Nature", "Sporty"]);

  // Disruptions: e.g. {1: "rainy", 2: "heat_wave"}
  const [disruptions, setDisruptions] = useState({});

  // Replan method: "quick" or "alns"
  const [replanMethod, setReplanMethod] = useState("quick");

  // Which day to display
  const [selectedDay, setSelectedDay] = useState("all");

  //----------------------------------------------------------
  // 2) On mount, load an initial solution.json if it exists
  //----------------------------------------------------------
  useEffect(() => {
    fetch('/solution.json')
      .then(res => {
        if (!res.ok) {
          // If there's no solution.json or the server returns 404/500
          throw new Error("No solution.json found or fetch error");
        }
        return res.json();
      })
      .then(data => {
        console.log("Loaded default solution.json:", data);
        setSolution(data);
      })
      .catch(err => {
        console.warn("Fetch solution.json error:", err);
        // Provide a minimal placeholder solution so the map doesn't break
        const placeholder = {
          tourist: {
            id: "placeholder",
            budget: 0,
            hotelLat,
            hotelLng: hotelLong
          },
          itinerary: []
        };
        setSolution(placeholder);
      });
  }, [hotelLat, hotelLong]);

  //----------------------------------------------------------
  // 3) If solution is null, show a loading message
  //----------------------------------------------------------
  if (!solution) {
    return <div>Loading initial route...</div>;
  }

  //----------------------------------------------------------
  // 4) Filter the itinerary by selectedDay for display
  //----------------------------------------------------------
  let itineraryToDisplay;
  if (selectedDay === "all") {
    itineraryToDisplay = solution.itinerary;
  } else {
    itineraryToDisplay = solution.itinerary.filter(
      (dayObj) => String(dayObj.day) === String(selectedDay)
    );
  }

  //----------------------------------------------------------
  // 5) Summaries: total cost, # attractions for current day(s)
  //----------------------------------------------------------
  const totalCost = itineraryToDisplay.reduce((acc, dayObj) => {
    return acc + dayObj.attractions.reduce((s, a) => s + a.cost, 0);
  }, 0);

  const totalAttractions = itineraryToDisplay.reduce((acc, dayObj) => {
    return acc + dayObj.attractions.length;
  }, 0);

  const summaryData = {
    totalCost,
    totalAttractions
  };

  //----------------------------------------------------------
  // 6) The "Generate Route" callback
  //----------------------------------------------------------
  const onGenerateRoute = () => {
    // Convert user disruptions from 1-based to 0-based
    const disruptionsZeroBased = {};
    for (let d = 1; d <= days; d++) {
      disruptionsZeroBased[d - 1] = disruptions[d] || "normal";
    }

    // Build the param object for the server
    const params = {
      id: solution.tourist.id,   // or a new ID
      preferences,
      budget,
      days,
      touring_hours: touringHours,
      hotelLat,
      hotelLong,
      seed: 123,

      // Stage 2 fields
      disruptions: disruptionsZeroBased,
      replan_method: replanMethod
    };

    console.log("Sending to back end:", params);

    fetch('http://localhost:5000/reoptimize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    })
      .then(res => {
        if (!res.ok) {
          throw new Error(`Server error: ${res.status} ${res.statusText}`);
        }
        return res.json();
      })
      .then(data => {
        // This is the brand-new solution from Python
        console.log("Got new solution from Python:", data);
        // Overwrite the existing solution in local state
        setSolution(data);
      })
      .catch(err => {
        console.error("Error reoptimizing:", err);
        alert("Error reoptimizing route. Check console/logs.");
      });
  };

  //----------------------------------------------------------
  // 7) Render the layout: map on left, side panel on right
  //----------------------------------------------------------
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <Header />

      <div style={{ flex: 1, display: 'flex' }}>
        {/* Left side: the map */}
        <div style={{ flex: 1 }}>
          <MapView
            key={selectedDay}
            touristStart={solution.tourist}
            itinerary={itineraryToDisplay}
          />
        </div>

        {/* Right side: the side panel for user input */}
        <SidePanel
          // Basic ALNS parameters
          budget={budget}
          setBudget={setBudget}
          days={days}
          setDays={setDays}
          touringHours={touringHours}
          setTouringHours={setTouringHours}
          hotelLat={hotelLat}
          setHotelLat={setHotelLat}
          hotelLong={hotelLong}
          setHotelLong={setHotelLong}
          preferences={preferences}
          setPreferences={setPreferences}

          // Disruptions
          disruptions={disruptions}
          setDisruptions={setDisruptions}

          // Replan method
          replanMethod={replanMethod}
          setReplanMethod={setReplanMethod}

          // Day-based display
          selectedDay={selectedDay}
          setSelectedDay={setSelectedDay}

          // Itinerary summary
          summary={summaryData}
          itinerary={itineraryToDisplay}

          // The callback to generate / reoptimize the route
          onGenerateRoute={onGenerateRoute}
        />
      </div>
    </div>
  );
}

export default App;
