// src/components/MapView.js
import React from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Fix Leaflet marker icons
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

const MapView = ({ touristStart, itinerary }) => {
  const singaporeCenter = [1.3521, 103.8198];

  const routeCoordinates = [[touristStart.hotelLat, touristStart.hotelLng]];
  const markers = [];

  // Add start marker (Hotel)
  markers.push(
    <Marker key="start" position={[touristStart.hotelLat, touristStart.hotelLng]}>
      <Popup>Hotel (Start Point)</Popup>
    </Marker>
  );

  // Add attraction markers and route
  itinerary.forEach((dayObj) => {
    dayObj.attractions.forEach((attr, idx) => {
      routeCoordinates.push([attr.lat, attr.lng]);

      const markerKey = `day${dayObj.day}-attr${idx}`;
      markers.push(
        <Marker key={markerKey} position={[attr.lat, attr.lng]}>
          <Popup>
            <strong>{attr.name}</strong>
            <br />
            Time: {attr.time}
            <br />
            Cost: {attr.cost}
            <br />
            Day: {dayObj.day}
          </Popup>
        </Marker>
      );
    });
  });

  return (
    <div style={{ height: '100%', width: '100%' }}>
      <MapContainer center={singaporeCenter} zoom={11} style={{ height: '100%', width: '100%' }}>
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution="&copy; OpenStreetMap contributors"
        />
        {markers}
        <Polyline positions={routeCoordinates} color="blue" />
      </MapContainer>
    </div>
  );
};

export default MapView;
