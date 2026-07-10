import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class MapScreen extends StatefulWidget {
  const MapScreen({super.key});

  @override
  State<MapScreen> createState() => _MapScreenState();
}

class _MapScreenState extends State<MapScreen> {
  Set<Marker> _markers = {};
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _fetchLatestLocation();
  }

  // 💡 Database එකේ ඇති marker_color අගය අනුව පැහැදිලි වර්ණ තෝරාගැනීම bn
  double _getMarkerHue(String? colorStr) {
    if (colorStr == "GREEN") return BitmapDescriptor.hueGreen;   // Validated (නිවැරදියි)
    if (colorStr == "RED") return BitmapDescriptor.hueRed;       // Rejected (වැරදියි)
    return BitmapDescriptor.hueOrange;                           // Pending / Disease (තැඹිලි)
  }

  Future<void> _fetchLatestLocation() async {
    try {
      final response = await http.get(Uri.parse('http://192.168.8.163:5000/get_latest_location'));

      if (response.statusCode == 200) {
        final List<dynamic> data = json.decode(response.body);
        Set<Marker> tempMarkers = {};
        
        if (data.isNotEmpty) {
          for (var item in data) {
            double lat = double.tryParse(item['latitude'].toString()) ?? 0.0;
            double lng = double.tryParse(item['longitude'].toString()) ?? 0.0;
            String? mColor = item['marker_color'];
            String validationStatus = item['status'] ?? "Pending";
            
            String predictionLabel = item['prediction'] ?? "Unknown";
            if (predictionLabel == "Nitrogen_Deficiency") predictionLabel = "නයිට්‍රජන් ඌණතාවය (N)";
            if (predictionLabel == "Phosphorus_Deficiency") predictionLabel = "පොස්පරස් ඌණතාවය (P)";
            if (predictionLabel == "Potassium_Deficiency") predictionLabel = "පොටෑසියම් ඌණතාවය (K)";

            tempMarkers.add(
              Marker(
                markerId: MarkerId(item['id'].toString()),
                position: LatLng(lat, lng),
                icon: BitmapDescriptor.defaultMarkerWithHue(_getMarkerHue(mColor)),
                infoWindow: InfoWindow(
                  title: "AI නිගමනය: $predictionLabel",
                  snippet: "තත්ත්වය: $validationStatus | විශ්වාසය: ${((item['confidence'] ?? 0.0) * 100).toStringAsFixed(1)}%",
                ),
              )
            );
          }

          setState(() {
            _markers = tempMarkers;
            _isLoading = false;
          });
        } else {
          setState(() { _isLoading = false; });
        }
      }
    } catch (e) {
      debugPrint("❌ Map Error: $e");
      setState(() { _isLoading = false; });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Current Analysis Map"),
        backgroundColor: Colors.green[700],
        foregroundColor: Colors.white,
      ),
      body: _isLoading 
          ? const Center(child: CircularProgressIndicator())
          : GoogleMap(
              initialCameraPosition: CameraPosition(
                target: _markers.isNotEmpty 
                    ? _markers.first.position 
                    : const LatLng(7.8731, 80.7718),
                zoom: 16, 
              ),
              markers: _markers,
              mapType: MapType.hybrid, 
              myLocationEnabled: true,
            ),
    );
  }
}