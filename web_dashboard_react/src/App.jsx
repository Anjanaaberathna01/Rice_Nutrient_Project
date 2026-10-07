import React, { useEffect, useState } from "react";
import axios from "axios";
import { MapContainer, TileLayer, CircleMarker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "./App.css";

function App() {
  const [locations, setLocations] = useState([]);
  const [loading, setLoading] = useState(true);

  const [soilData, setSoilData] = useState([]);
  const [formData, setFormData] = useState({
    ds_division: "",
    gn_division: "",
    ph_value: "",
    ec_value: "",
    phosphorus_value: "",
    potassium_value: "",
    organic_matter_status: ""
  });

  const [editingId, setEditingId] = useState(null);

  const API_URL = "http://172.20.10.8:5000";

  useEffect(() => {
    fetchLocations();
    fetchSoilData();
  }, []);

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

  const getMarkerColor = (colorCode) => {
    switch (colorCode) {
      case "GREEN":
        return "#28a745";
      case "RED":
        return "#dc3545";
      case "ORANGE":
      default:
        return "#fd7e14";
    }
  };

  const diseases = locations.filter(
    (loc) => loc.category === "Disease"
  );

  const deficiencies = locations.filter(
    (loc) => loc.category === "Deficiency"
  );

  const defGreen = deficiencies.filter(
    (loc) => loc.marker_color === "GREEN"
  );

  const defRed = deficiencies.filter(
    (loc) => loc.marker_color === "RED"
  );

  const defOrange = deficiencies.filter(
    (loc) => loc.marker_color === "ORANGE" || !loc.marker_color
  );

  const RecordCard = ({ loc }) => (
    <div
      className="record-card"
      style={{
        borderLeftColor: getMarkerColor(loc.marker_color)
      }}
    >
      <div className="record-title">
        {loc.prediction}
      </div>

      <div className="record-meta">
        Status: <b>{loc.status}</b>
      </div>

      <div className="record-meta">
        Confidence:{" "}
        {(parseFloat(loc.confidence) * 100).toFixed(1)}%
      </div>
    </div>
  );

  // =========================
  // SOIL DATA CRUD
  // =========================

  const handleFormChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleCsvSubmit = async (e) => {
    e.preventDefault();

    try {
      if (editingId !== null) {
        await axios.put(
          `${API_URL}/api/soil/${editingId}`,
          formData
        );
        alert("Soil record updated successfully!");
      } else {
        await axios.post(
          `${API_URL}/api/soil`,
          formData
        );
        alert("New soil record added successfully!");
      }

      setFormData({
        ds_division: "",
        gn_division: "",
        ph_value: "",
        ec_value: "",
        phosphorus_value: "",
        potassium_value: "",
        organic_matter_status: ""
      });

      setEditingId(null);
      fetchSoilData();

    } catch (error) {
      console.error("Error saving soil data:", error);
      alert("Error processing your request.");
    }
  };

  const editRow = (row) => {
    setFormData({
      ds_division: row.ds_division,
      gn_division: row.gn_division,
      ph_value: row.ph_value,
      ec_value: row.ec_value,
      phosphorus_value: row.phosphorus_value,
      potassium_value: row.potassium_value,
      organic_matter_status: row.organic_matter_status
    });

    setEditingId(row.id);

    window.scrollTo({
      top: document.body.scrollHeight / 2,
      behavior: "smooth"
    });
  };

  const deleteRow = async (id) => {
    if (
      window.confirm(
        "Are you sure you want to delete this soil record?"
      )
    ) {
      try {
        await axios.delete(
          `${API_URL}/api/soil/${id}`
        );
        fetchSoilData();
      } catch (error) {
        console.error("Error deleting soil data:", error);
      }
    }
  };

  return (
    <div className="admin-page">

      <style>{`

        * {
          box-sizing: border-box;
        }

        .admin-page {
          font-family: Arial, Helvetica, sans-serif;
          background: #f4f7f6;
          min-height: 100vh;
          padding-bottom: 50px;
          color: #333;
        }

        /* =========================
           HEADER
        ========================= */

        .admin-header {
          background-image:
            linear-gradient(
              rgba(0,0,0,.62),
              rgba(0,0,0,.62)
            ),
            url("/rice_bg.jpg");

          background-size: cover;
          background-position: center;

          min-height: 280px;

          display: flex;
          align-items: center;
          justify-content: center;

          text-align: center;

          padding: 45px 20px;

          color: white;

          box-shadow:
            0 4px 10px rgba(0,0,0,.12);
        }

        .admin-header h1 {
          margin: 0;

          font-size:
            clamp(32px, 5vw, 48px);

          font-weight: 700;

          letter-spacing: .5px;
        }

        .admin-header p {
          margin-top: 12px;

          font-size:
            clamp(15px, 2vw, 18px);

          opacity: .92;
        }

        /* =========================
           MAIN CONTAINER
        ========================= */

        .admin-container {
          width:
            min(1400px, calc(100% - 40px));

          margin:
            30px auto 0;
        }

        /* =========================
           SECTION CARD
        ========================= */

        .section-card {
          background: #fff;

          border-radius: 10px;

          padding: 28px;

          margin-bottom: 30px;

          box-shadow:
            0 2px 8px rgba(0,0,0,.08);

          overflow: hidden;
        }

        .section-title {
          margin: 0 0 22px;

          padding-bottom: 12px;

          border-bottom:
            2px solid #28a745;

          color: #333;

          font-size: 24px;

          font-weight: 700;
        }

        .section-title.blue {
          border-bottom-color: #0056b3;
        }

        /* =========================
           MAP SECTION
        ========================= */

        .map-layout {
          display: grid;

          grid-template-columns:
            minmax(260px, 350px)
            minmax(0, 1fr);

          gap: 20px;

          height: 65vh;

          min-height: 500px;
        }

        .records-panel {
          min-width: 0;

          overflow-y: auto;

          padding-right: 8px;
        }

        .records-heading {
          margin: 0 0 10px;

          padding-bottom: 7px;

          font-size: 17px;

          border-bottom:
            2px solid currentColor;
        }

        .records-heading.orange {
          color: #fd7e14;
        }

        .records-heading.blue {
          color: #0056b3;
        }

        .records-heading.green {
          color: #28a745;
        }

        .records-heading.red {
          color: #dc3545;
        }

        .record-card {
          border-left:
            5px solid #28a745;

          border-radius: 6px;

          padding: 11px;

          margin-bottom: 10px;

          background: #fff;

          box-shadow:
            0 1px 4px rgba(0,0,0,.09);
        }

        .record-title {
          font-weight: 700;

          font-size: 14px;
        }

        .record-meta {
          font-size: 12px;

          color: #666;

          margin-top: 4px;
        }

        .map-wrapper {
          min-width: 0;

          height: 100%;

          border:
            2px solid #ddd;

          border-radius: 8px;

          overflow: hidden;

          z-index: 1;
        }

        /* =========================
           FORMS
        ========================= */

        .data-form {
          display: grid;

          gap: 15px;

          margin-bottom: 25px;

          background: #f8f9fa;

          padding: 20px;

          border-radius: 8px;
        }

        .soil-form {
          grid-template-columns:
            repeat(3, minmax(0, 1fr));
        }

        .form-control {
          width: 100%;

          min-width: 0;

          padding: 11px 12px;

          border-radius: 5px;

          border:
            1px solid #ccc;

          background: #fff;

          font-size: 14px;

          outline: none;
        }

        .form-control:focus {
          border-color: #0056b3;

          box-shadow:
            0 0 0 2px
            rgba(0,86,179,.12);
        }

        .form-actions {
          grid-column: 1 / -1;

          display: flex;

          justify-content: flex-end;

          align-items: center;

          gap: 10px;

          flex-wrap: wrap;
        }

        /* =========================
           BUTTONS
        ========================= */

        .btn {
          border: none;

          padding: 10px 18px;

          border-radius: 5px;

          cursor: pointer;

          font-weight: 700;

          font-size: 14px;

          transition:
            opacity .15s ease,
            transform .15s ease;
        }

        .btn:hover {
          opacity: .9;

          transform:
            translateY(-1px);
        }

        .btn-add {
          background: #0d6efd;
          color: #fff;
        }

        .btn-update {
          background: #ffc107;
          color: #000;
        }

        .btn-cancel {
          background: #6c757d;
          color: #fff;
        }

        .btn-edit {
          background: #1ab817;
          color: #fff;

          padding: 7px 12px;
        }

        .btn-delete {
          background: #35dc64;
          color: #fff;

          padding: 7px 12px;
        }

        /* =========================
           TABLE
        ========================= */

        .table-wrapper {
          width: 100%;

          max-height: 350px;

          overflow: auto;

          border:
            1px solid #ddd;

          border-radius: 5px;
        }

        .data-table {
          width: 100%;

          min-width: 850px;

          border-collapse:
            collapse;

          text-align: left;

          font-size: 14px;
        }

        .data-table th {
          padding: 12px;

          background: #343a40;

          color: #fff;

          position: sticky;

          top: 0;

          z-index: 2;

          white-space: nowrap;
        }

        .data-table td {
          padding: 12px;

          border-bottom:
            1px solid #ddd;

          white-space: nowrap;
        }

        .data-table tbody tr:nth-child(even) {
          background: #f8f9fa;
        }

        .data-table tbody tr:nth-child(odd) {
          background: #fff;
        }

        .action-cell {
          text-align: center;
        }

        .empty-text {
          font-size: 12px;

          color: #777;
        }

        /* =========================
           TABLET
        ========================= */

        @media (max-width: 1000px) {

          .map-layout {
            grid-template-columns: 1fr;
            height: auto;
          }

          .records-panel {
            max-height: 360px;
          }

          .map-wrapper {
            height: 520px;
          }

          .soil-form {
            grid-template-columns:
              repeat(2, minmax(0, 1fr));
          }
        }

        /* =========================
           MOBILE
        ========================= */

        @media (max-width: 650px) {

          .admin-container {
            width:
              min(100% - 20px, 1400px);

            margin-top: 15px;
          }

          .admin-header {
            min-height: 220px;

            padding:
              35px 15px;
          }

          .section-card {
            padding: 18px;

            margin-bottom: 18px;

            border-radius: 8px;
          }

          .section-title {
            font-size: 20px;
          }

          .soil-form {
            grid-template-columns: 1fr;
            padding: 15px;
          }

          .form-actions {
            justify-content: stretch;
          }

          .form-actions .btn {
            flex: 1;
            min-width: 120px;
          }

          .map-wrapper {
            height: 430px;
          }

          .records-panel {
            max-height: 330px;
          }

          .data-table {
            min-width: 850px;
          }
        }

      `}</style>

      {/* =========================
          HEADER
      ========================= */}

      <header className="admin-header">
        <div>
          <h1>Admin Panel</h1>
          <p>
          </p>
        </div>
      </header>

      <main className="admin-container">

        {/* =========================
            LIVE FIELD TRIANGULATION
        ========================= */}

        <section className="section-card">
          <h2 className="section-title">
            Live Field Triangulation
          </h2>

          {loading ? (
            <p>Loading Map Data...</p>
          ) : (
            <div className="map-layout">

              {/* RECORDS */}
              <div className="records-panel">
                <h3 className="records-heading orange">
                  Diseases (Pending) - {diseases.length}
                </h3>
                {diseases.length === 0 ? (
                  <p className="empty-text">No disease records found.</p>
                ) : (
                  diseases.map((loc) => (
                    <RecordCard key={loc.id} loc={loc} />
                  ))
                )}

                <h3 className="records-heading blue" style={{ marginTop: 25 }}>
                  Deficiencies - {deficiencies.length}
                </h3>

                <h4 className="records-heading green" style={{ marginTop: 10 }}>
                  Validated (Urea/TSP/MOP) - {defGreen.length}
                </h4>
                {defGreen.length === 0 ? (
                  <p className="empty-text">No validated records.</p>
                ) : (
                  defGreen.map((loc) => (
                    <RecordCard key={loc.id} loc={loc} />
                  ))
                )}

                <h4 className="records-heading red" style={{ marginTop: 15 }}>
                  Rejected (Safe) - {defRed.length}
                </h4>
                {defRed.length === 0 ? (
                  <p className="empty-text">No rejected records.</p>
                ) : (
                  defRed.map((loc) => (
                    <RecordCard key={loc.id} loc={loc} />
                  ))
                )}

                <h4 className="records-heading orange" style={{ marginTop: 15 }}>
                  Pending (No Soil Data) - {defOrange.length}
                </h4>
                {defOrange.length === 0 ? (
                  <p className="empty-text">No pending records.</p>
                ) : (
                  defOrange.map((loc) => (
                    <RecordCard key={loc.id} loc={loc} />
                  ))
                )}
              </div>

              {/* MAP */}
              <div className="map-wrapper">
                <MapContainer
                  center={[8.3114, 80.4037]}
                  zoom={10}
                  style={{ height: "100%", width: "100%" }}
                >
                  <TileLayer
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    attribution="&copy; OpenStreetMap"
                  />
                  {locations.map((loc) => (
                    <CircleMarker
                      key={loc.id}
                      center={[
                        parseFloat(loc.latitude),
                        parseFloat(loc.longitude)
                      ]}
                      radius={8}
                      pathOptions={{
                        fillColor: getMarkerColor(loc.marker_color),
                        color: "#fff",
                        weight: 2,
                        fillOpacity: 0.9
                      }}
                    >
                      <Popup>
                        <div style={{ fontSize: "14px" }}>
                          <b>Prediction:</b> {loc.prediction}
                          <br />
                          <b>Status:</b> {loc.status}
                          <br />
                          <b>Confidence:</b>{" "}
                          {(parseFloat(loc.confidence) * 100).toFixed(1)}%
                        </div>
                      </Popup>
                    </CircleMarker>
                  ))}
                </MapContainer>
              </div>
            </div>
          )}
        </section>

        {/* =========================
            SOIL DATA MANAGEMENT
        ========================= */}

        <section className="section-card">
          <h2 className="section-title blue">
            Soil Data Management
          </h2>

          <form onSubmit={handleCsvSubmit} className="data-form soil-form">
            <input
              className="form-control"
              type="text"
              name="ds_division"
              placeholder="DS Division"
              value={formData.ds_division}
              onChange={handleFormChange}
              required
            />
            <input
              className="form-control"
              type="text"
              name="gn_division"
              placeholder="GN Division"
              value={formData.gn_division}
              onChange={handleFormChange}
              required
            />
            <input
              className="form-control"
              type="number"
              step="0.1"
              name="ph_value"
              placeholder="pH Value"
              value={formData.ph_value}
              onChange={handleFormChange}
              required
            />
            <input
              className="form-control"
              type="number"
              step="0.001"
              name="ec_value"
              placeholder="EC Value"
              value={formData.ec_value}
              onChange={handleFormChange}
              required
            />
            <input
              className="form-control"
              type="number"
              step="0.1"
              name="phosphorus_value"
              placeholder="Phosphorus (mg/kg)"
              value={formData.phosphorus_value}
              onChange={handleFormChange}
              required
            />
            <input
              className="form-control"
              type="number"
              step="0.1"
              name="potassium_value"
              placeholder="Potassium (mg/kg)"
              value={formData.potassium_value}
              onChange={handleFormChange}
              required
            />
            <input
              className="form-control"
              type="number"
              step="0.1"
              name="organic_matter_status"
              placeholder="Organic Matter (%)"
              value={formData.organic_matter_status}
              onChange={handleFormChange}
              required
            />

            <div className="form-actions">
              <button
                type="submit"
                className={`btn ${editingId !== null ? "btn-update" : "btn-add"}`}
              >
                {editingId !== null ? "Update Record" : "Add Soil Record"}
              </button>

              {editingId !== null && (
                <button
                  type="button"
                  className="btn btn-cancel"
                  onClick={() => {
                    setEditingId(null);
                    setFormData({
                      ds_division: "",
                      gn_division: "",
                      ph_value: "",
                      ec_value: "",
                      phosphorus_value: "",
                      potassium_value: "",
                      organic_matter_status: ""
                    });
                  }}
                >
                  Cancel
                </button>
              )}
            </div>
          </form>

          <div className="table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>DS Division</th>
                  <th>GN Division</th>
                  <th>pH</th>
                  <th>EC</th>
                  <th>Phosphorus</th>
                  <th>Potassium</th>
                  <th>Organic Matter</th>
                  <th className="action-cell">Actions</th>
                </tr>
              </thead>
              <tbody>
                {soilData.map((row, index) => (
                  <tr key={index}>
                    <td>{row.ds_division}</td>
                    <td>{row.gn_division}</td>
                    <td>{row.ph_value}</td>
                    <td>{row.ec_value}</td>
                    <td>{row.phosphorus_value}</td>
                    <td>{row.potassium_value}</td>
                    <td>{row.organic_matter_status}</td>
                    <td className="action-cell">
                      <button
                        onClick={() => editRow(row)}
                        className="btn btn-edit"
                      >
                        Edit
                      </button>
                      <button
                        onClick={() => deleteRow(row.id)}
                        className="btn btn-delete"
                        style={{ marginLeft: 5 }}
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

      </main>
    </div>
  );
}

export default App;