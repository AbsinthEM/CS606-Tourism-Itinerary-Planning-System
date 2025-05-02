// src/components/SidePanel.js
import React from 'react';

// A helper function to parse "start-end" strings (e.g. "10.0-11.0") and return a duration.
function parseDuration(timeStr) {
  if (!timeStr) return null;
  const [start, end] = timeStr.split('-').map((t) => parseFloat(t));
  if (isNaN(start) || isNaN(end)) return null;
  return end - start; // e.g. 11.0 - 10.0 = 1
}

const SidePanel = ({
  // Props for ALNS parameters
  budget,
  setBudget,
  days,
  setDays,
  touringHours,
  setTouringHours,
  hotelLat,
  setHotelLat,
  hotelLong,
  setHotelLong,
  preferences,
  setPreferences,

  // Disruptions
  disruptions,
  setDisruptions,

  // Replan method
  replanMethod,
  setReplanMethod,

  // For day-based display
  selectedDay,
  setSelectedDay,

  // Itinerary summary
  summary = {},
  itinerary = [],

  // Callback to run the route generation
  onGenerateRoute
}) => {

  // For each day, show a disruption dropdown
  const handleDisruptionChange = (dayIndex, newValue) => {
    setDisruptions((prev) => ({
      ...prev,
      [dayIndex]: newValue
    }));
  };

  return (
    <div style={{ width: '300px', background: '#BBACC1', padding: '10px', borderLeft: '1px solid #ccc' }}>
      <h2>Parameters</h2>

      {/* Budget */}
      <div style={{ marginBottom: '10px' }}>
        <label>Budget: </label>
        <input
          type="number"
          value={budget}
          onChange={(e) => setBudget(Number(e.target.value))}
          style={{ padding: '5px' }}
        />
      </div>

      {/* Replan Method */}
      <div style={{ marginBottom: '10px' }}>
        <label>Replan Method: </label>
        <select
          value={replanMethod}
          onChange={(e) => setReplanMethod(e.target.value)}
          style={{ padding: '5px' }}
        >
          <option value="quick">Quick Repair</option>
          <option value="alns">ALNS Replan</option>
        </select>
      </div>

      {/* Day selector for itinerary display */}
      <div style={{ marginBottom: '10px' }}>
        <label>Display Day: </label>
        <select
          value={selectedDay}
          onChange={(e) => setSelectedDay(e.target.value)}
          style={{ padding: '5px' }}
        >
          <option value="all">All</option>
          {Array.from({ length: days }, (_, i) => i + 1).map((dayNum) => (
            <option key={dayNum} value={String(dayNum)}>
              Day {dayNum}
            </option>
          ))}
        </select>
      </div>

      <hr />

      <h2>Tourist Profile</h2>

      {/* Days */}
      <div style={{ marginBottom: '10px' }}>
        <label style={{ display: 'block', marginBottom: '5px' }}>Days:</label>
        <input
          type="number"
          value={days}
          onChange={(e) => setDays(Number(e.target.value))}
          style={{ padding: '5px' }}
        />
      </div>

      {/* Touring Hours */}
      <div style={{ marginBottom: '10px' }}>
        <label style={{ display: 'block', marginBottom: '5px' }}>Touring Hours (start,end):</label>
        <input
          type="text"
          value={touringHours}
          onChange={(e) => setTouringHours(e.target.value.split(',').map(item => Number(item.trim())))}
          placeholder="e.g., 8,19"
          style={{ padding: '5px' }}
        />
      </div>

      {/* Hotel Coordinates */}
      <div style={{ marginBottom: '10px' }}>
        <label style={{ display: 'block', marginBottom: '5px' }}>Hotel Location Latitude:</label>
        <input
          type="number"
          value={hotelLat}
          onChange={(e) => setHotelLat(Number(e.target.value))}
          style={{ padding: '5px' }}
        />
      </div>
      <div style={{ marginBottom: '10px' }}>
        <label style={{ display: 'block', marginBottom: '5px' }}>Hotel Location Longitude:</label>
        <input
          type="number"
          value={hotelLong}
          onChange={(e) => setHotelLong(Number(e.target.value))}
          style={{ padding: '5px' }}
        />
      </div>

      {/* Preferences */}
      <div style={{ marginBottom: '10px' }}>
        <label style={{ display: 'block', marginBottom: '5px' }}>Preferences (comma separated):</label>
        <input
          type="text"
          // value={preferences.join(',')}
          value={preferences ? preferences.join(',') : ''}
          onChange={(e) => setPreferences(e.target.value.split(',').map(item => item.trim()))}
          placeholder="e.g., Nature,Sporty"
          style={{ padding: '5px' }}
        />
      </div>

      {/* Disruptions for each day */}
      <h2>Disruptions</h2>
      {Array.from({ length: days }, (_, i) => i + 1).map((dayNum) => {
        const currentValue = disruptions[dayNum] || "normal";
        return (
          <div key={dayNum} style={{ marginBottom: '10px' }}>
            <label>Day {dayNum} Disruption: </label>
            <select
              value={currentValue}
              onChange={(e) => handleDisruptionChange(dayNum, e.target.value)}
              style={{ padding: '5px', marginLeft: '5px' }}
            >
              <option value="normal">Normal</option>
              <option value="rainy">Rainy</option>
              <option value="heat_wave">Heat Wave</option>
              <option value="early_closure">Early Closure</option>
              <option value="family_day">Family Day</option>
            </select>
          </div>
        );
      })}

      {/* Generate Route Button */}
      <button 
        onClick={onGenerateRoute} 
        style={{
          padding: '10px',
          marginBottom: '10px',
          backgroundColor: '#392A16',
          color: '#fff',
          border: 'none',
          cursor: 'pointer',
          width: '100%',
          fontSize: '18px'
        }}
      >
        Generate Route
      </button>

      <hr />

      <h2>Itinerary Summary</h2>
      <p><strong>Total Cost:</strong> SGD {summary.totalCost}</p>
      <p><strong>Attractions:</strong> {summary.totalAttractions}</p>

      {/* Detailed breakdown by day and attraction */}
      {itinerary.map((dayObj) => (
        <div key={dayObj.day} style={{ marginTop: '10px' }}>
          <strong>Day {dayObj.day}</strong>
          <ul style={{ listStyleType: 'disc', marginLeft: '20px' }}>
            {dayObj.attractions.map((attr, idx) => {
              const duration = parseDuration(attr.time); 
              return (
                <li key={idx}>
                  {attr.name}
                  {duration !== null && ` (${duration} hour${duration > 1 ? 's' : ''})`}
                </li>
              );
            })}
          </ul>
        </div>
      ))}
    </div>
  );
};

export default SidePanel;
