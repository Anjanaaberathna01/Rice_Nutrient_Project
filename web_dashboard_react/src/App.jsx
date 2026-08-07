import React, { useEffect, useState } from "react";
import axios from "axios";
import { MapContainer, TileLayer, CircleMarker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "./App.css";

function App() {
  // 💡 Map States
  const [locations, setLocations] = useState([]);
  const [loading, setLoading] = useState(true);

  // 💡 CSV Data States
  const [soilData, setSoilData] = useState([]);
  const [formData, setFormData] = useState({
    ds_division: "", 
    gn_division: "", 
    ph_value: "", 
    phosphorus_value: "", 
    potassium_value: "", 
    organic_matter_status: ""
  });
  const [editingId, setEditingId] = useState(null);

  // Flask IP එක
  const API_URL = "http://192.168.8.229:5000";

  useEffect(() => {
    fetchLocations();
    fetchSoilData();
  }, []);

  // --- API Functions ---
  const fetchLocations = async () => {
    try {
      const response = await axios.get(`${API_URL}/get_latest_location`);
      setLocations(response.data);
      setLoading(false);
    } catch (error) {
      console.error("Error fetching map data:", error);
      setLoading(false);
    }
  };

  const fetchSoilData = async () => {
    try {
      const res = await axios.get(`${API_URL}/api/soil`);
      setSoilData(res.data);
    } catch (error) {
      console.error("Error fetching soil data:", error);
    }
  };

  // --- Map Logic ---
  const getMarkerColor = (colorCode) => {
    switch (colorCode) {
      case "GREEN": return "#28a745";
      case "RED": return "#dc3545";
      case "ORANGE": default: return "#fd7e14";
    }
  };

  const diseases = locations.filter((loc) => loc.category === "Disease");
  const deficiencies = locations.filter((loc) => loc.category === "Deficiency");
  const defGreen = deficiencies.filter((loc) => loc.marker_color === "GREEN");
  const defRed = deficiencies.filter((loc) => loc.marker_color === "RED");
  const defOrange = deficiencies.filter((loc) => loc.marker_color === "ORANGE" || !loc.marker_color);

  const RecordCard = ({ loc }) => (
    <div style={{
      borderLeft: `5px solid ${getMarkerColor(loc.marker_color)}`,
      borderRadius: "6px", padding: "10px", marginBottom: "10px", backgroundColor: "#fff", boxShadow: "0 1px 3px rgba(0,0,0,0.1)"
    }}>
      <div style={{ fontWeight: "bold", fontSize: "14px", color: "#333" }}>{loc.prediction}</div>
      <div style={{ fontSize: "12px", color: "#666", marginTop: "4px" }}>Status: <b>{loc.status}</b></div>
      <div style={{ fontSize: "12px", color: "#666" }}>Confidence: {(parseFloat(loc.confidence) * 100).toFixed(1)}%</div>
    </div>
  );

  // --- CSV CRUD Logic ---
  const handleFormChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleCsvSubmit = async (e) => {
    e.preventDefault();
    try {
      if (editingId !== null) {
        await axios.put(`${API_URL}/api/soil/${editingId}`, formData);
        alert("Soil record updated successfully! ✅");
      } else {
        await axios.post(`${API_URL}/api/soil`, formData);
        alert("New soil record added successfully! ✅");
      }
      setFormData({ ds_division: "", gn_division: "", ph_value: "", phosphorus_value: "", potassium_value: "", organic_matter_status: "" });
      setEditingId(null);
      fetchSoilData();
    } catch (error) {
      console.error("Error saving CSV data:", error);
      alert("Error processing your request.");
    }
  };

  const editRow = (row) => {
    setFormData({
      ds_division: row.ds_division, gn_division: row.gn_division, ph_value: row.ph_value, 
      phosphorus_value: row.phosphorus_value, potassium_value: row.potassium_value, organic_matter_status: row.organic_matter_status
    });
    setEditingId(row.id);
    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' }); // පහළට යනවා ඔටෝමැටික්
  };

  const deleteRow = async (id) => {
    if (window.confirm("Are you sure you want to delete this soil record? ❌")) {
      try {
        await axios.delete(`${API_URL}/api/soil/${id}`);
        fetchSoilData();
      } catch (error) {
        console.error("Error deleting data:", error);
      }
    }
  };

  return (
    <div style={{ fontFamily: "sans-serif", backgroundColor: "#f4f7f6", minHeight: "100vh", paddingBottom: "50px" }}>
      
      {/* 1. HEADER SECTION (Background Image) */}
      <div style={{ 
        backgroundImage: 'linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), url("/rice_bg.jpg")', 
        backgroundSize: "cover", 
        backgroundPosition: "center", 
        color: "white", 
        padding: "60px 20px", 
        textAlign: "center",
        boxShadow: "0 4px 6px rgba(0,0,0,0.1)"
      }}>
        <h1 style={{ margin: 0, fontSize: "42px", letterSpacing: "1px" }}>Precision Agriculture System</h1>
        <p style={{ marginTop: "10px", fontSize: "18px", opacity: 0.9 }}>Admin Dashboard: Real-time Triangulation & Soil Analytics</p>
      </div>

      <div style={{ padding: "30px", maxWidth: "1400px", margin: "0 auto" }}>
        
        {/* 2. DASHBOARD SECTION (Map + Sidebar) */}
        <div style={{ backgroundColor: "#fff", padding: "20px", borderRadius: "10px", boxShadow: "0 2px 8px rgba(0,0,0,0.1)", marginBottom: "40px" }}>
          <h2 style={{ borderBottom: "2px solid #28a745", paddingBottom: "10px", color: "#333" }}>📍 Live Field Triangulation</h2>
          {loading ? (
            <p>Loading Map Data...</p>
          ) : (
            <div style={{ display: "flex", gap: "20px", height: "65vh" }}>
              
              {/* Sidebar */}
              <div style={{ width: "350px", overflowY: "auto", paddingRight: "10px" }}>
                <h3 style={{ color: "#fd7e14", borderBottom: "2px solid #fd7e14", paddingBottom: "5px" }}>🦠 Diseases (Pending) - {diseases.length}</h3>
                {diseases.length === 0 ? <p style={{fontSize:"12px"}}>No disease records found.</p> : diseases.map(loc => <RecordCard key={loc.id} loc={loc} />)}

                <h3 style={{ color: "#0056b3", borderBottom: "2px solid #0056b3", paddingBottom: "5px", marginTop: "25px" }}>🧪 Deficiencies - {deficiencies.length}</h3>
                <h4 style={{ color: "#28a745", margin: "10px 0 5px 0" }}>✅ Validated (Urea/TSP/MOP) - {defGreen.length}</h4>
                {defGreen.length === 0 ? <p style={{fontSize:"12px"}}>No validated records.</p> : defGreen.map(loc => <RecordCard key={loc.id} loc={loc} />)}
                
                <h4 style={{ color: "#dc3545", margin: "15px 0 5px 0" }}>❌ Rejected (Safe) - {defRed.length}</h4>
                {defRed.length === 0 ? <p style={{fontSize:"12px"}}>No rejected records.</p> : defRed.map(loc => <RecordCard key={loc.id} loc={loc} />)}
                
                <h4 style={{ color: "#fd7e14", margin: "15px 0 5px 0" }}>⚠️ Pending (No Soil Data) - {defOrange.length}</h4>
                {defOrange.length === 0 ? <p style={{fontSize:"12px"}}>No pending records.</p> : defOrange.map(loc => <RecordCard key={loc.id} loc={loc} />)}
              </div>

              {/* Map */}
              <div style={{ flex: 1, border: "2px solid #ddd", borderRadius: "8px", overflow: "hidden", zIndex: 1 }}>
                <MapContainer center={[8.3114, 80.4037]} zoom={10} style={{ height: "100%", width: "100%" }}>
                  <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" attribution='&copy; OpenStreetMap' />
                  {locations.map((loc) => (
                    <CircleMarker key={loc.id} center={[parseFloat(loc.latitude), parseFloat(loc.longitude)]} radius={8} pathOptions={{ fillColor: getMarkerColor(loc.marker_color), color: "#fff", weight: 2, fillOpacity: 0.9 }}>
                      <Popup>
                        <div style={{ fontSize: "14px" }}>
                          <b>Prediction:</b> {loc.prediction} <br/>
                          <b>Status:</b> {loc.status} <br/>
                          <b>Confidence:</b> {(parseFloat(loc.confidence) * 100).toFixed(1)}% <br/>
                        </div>
                      </Popup>
                    </CircleMarker>
                  ))}
                </MapContainer>
              </div>
            </div>
          )}
        </div>

        {/* 3. CSV MANAGEMENT SECTION */}
        <div style={{ backgroundColor: "#fff", padding: "30px", borderRadius: "10px", boxShadow: "0 2px 8px rgba(0,0,0,0.1)" }}>
          <h2 style={{ borderBottom: "2px solid #0056b3", paddingBottom: "10px", color: "#333", marginBottom: "20px" }}>🗄️ Soil Data Management (CSV Database)</h2>
          
          {/* Form */}
          <form onSubmit={handleCsvSubmit} style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "15px", marginBottom: "30px", backgroundColor: "#f8f9fa", padding: "20px", borderRadius: "8px" }}>
            <input type="text" name="ds_division" placeholder="DS Division" value={formData.ds_division} onChange={handleFormChange} required style={{ padding: "10px", borderRadius: "5px", border: "1px solid #ccc" }} />
            <input type="text" name="gn_division" placeholder="GN Division" value={formData.gn_division} onChange={handleFormChange} required style={{ padding: "10px", borderRadius: "5px", border: "1px solid #ccc" }} />
            <input type="number" step="0.1" name="ph_value" placeholder="pH Value" value={formData.ph_value} onChange={handleFormChange} required style={{ padding: "10px", borderRadius: "5px", border: "1px solid #ccc" }} />
            <input type="number" step="0.1" name="phosphorus_value" placeholder="Phosphorus (mg/kg)" value={formData.phosphorus_value} onChange={handleFormChange} required style={{ padding: "10px", borderRadius: "5px", border: "1px solid #ccc" }} />
            <input type="number" step="0.1" name="potassium_value" placeholder="Potassium (mg/kg)" value={formData.potassium_value} onChange={handleFormChange} required style={{ padding: "10px", borderRadius: "5px", border: "1px solid #ccc" }} />
            <input type="number" step="0.1" name="organic_matter_status" placeholder="Organic Matter (%)" value={formData.organic_matter_status} onChange={handleFormChange} required style={{ padding: "10px", borderRadius: "5px", border: "1px solid #ccc" }} />
            
            <div style={{ gridColumn: "span 3", textAlign: "right" }}>
              <button type="submit" style={{ backgroundColor: editingId !== null ? "#ffc107" : "#0d6efd", color: editingId !== null ? "#000" : "#fff", padding: "10px 20px", border: "none", borderRadius: "5px", cursor: "pointer", fontWeight: "bold" }}>
                {editingId !== null ? "Update Record 🔄" : "Add New Record ➕"}
              </button>
              {editingId !== null && (
                <button type="button" onClick={() => { setEditingId(null); setFormData({ ds_division: "", gn_division: "", ph_value: "", phosphorus_value: "", potassium_value: "", organic_matter_status: "" }); }} style={{ backgroundColor: "#6c757d", color: "#fff", padding: "10px 20px", border: "none", borderRadius: "5px", cursor: "pointer", marginLeft: "10px", fontWeight: "bold" }}>
                  Cancel
                </button>
              )}
            </div>
          </form>

          {/* Table */}
          <div style={{ maxHeight: "400px", overflowY: "auto", border: "1px solid #ddd", borderRadius: "5px" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left" }}>
              <thead style={{ backgroundColor: "#343a40", color: "#fff", position: "sticky", top: 0 }}>
                <tr>
                  <th style={{ padding: "12px" }}>DS Division</th>
                  <th style={{ padding: "12px" }}>GN Division</th>
                  <th style={{ padding: "12px" }}>pH</th>
                  <th style={{ padding: "12px" }}>Phosphorus</th>
                  <th style={{ padding: "12px" }}>Potassium</th>
                  <th style={{ padding: "12px" }}>Organic Matter</th>
                  <th style={{ padding: "12px", textAlign: "center" }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {soilData.map((row, index) => (
                  <tr key={index} style={{ borderBottom: "1px solid #ddd", backgroundColor: index % 2 === 0 ? "#f8f9fa" : "#fff" }}>
                    <td style={{ padding: "12px" }}>{row.ds_division}</td>
                    <td style={{ padding: "12px" }}>{row.gn_division}</td>
                    <td style={{ padding: "12px" }}>{row.ph_value}</td>
                    <td style={{ padding: "12px" }}>{row.phosphorus_value}</td>
                    <td style={{ padding: "12px" }}>{row.potassium_value}</td>
                    <td style={{ padding: "12px" }}>{row.organic_matter_status}</td>
                    <td style={{ padding: "12px", textAlign: "center" }}>
                      <button onClick={() => editRow(row)} style={{ backgroundColor: "#17a2b8", color: "#fff", border: "none", padding: "5px 10px", borderRadius: "4px", cursor: "pointer", marginRight: "5px" }}>Edit</button>
                      <button onClick={() => deleteRow(row.id)} style={{ backgroundColor: "#dc3545", color: "#fff", border: "none", padding: "5px 10px", borderRadius: "4px", cursor: "pointer" }}>Delete</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

        </div>
      </div>
    </div>
  );
}

export default App;