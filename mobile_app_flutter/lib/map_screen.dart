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
  bool _isLoading = true; // දත්ත එනකම් පෙන්වන්න

  @override
  void initState() {
    super.initState();
    _fetchLocations();
  }

  Future<void> _fetchLocations() async {
    try {
      final response = await http.get(Uri.parse('http://192.168.8.163:5000/get_locations'));

      if (response.statusCode == 200) {
        final List<dynamic> data = json.decode(response.body);
        
        // Debugging සඳහා ලොග් එකක් දාමු
        print("Fetched Data: ${data.length} items found");

        final Set<Marker> tempMarkers = data.map((item) {
          // වැදගත්: Latitude/Longitude null හෝ වැරදි format නම් එය පාලනය කිරීම
          double lat = double.tryParse(item['latitude'].toString()) ?? 0.0;
          double lng = double.tryParse(item['longitude'].toString()) ?? 0.0;
          
          double markerColor;
          if (item['category'] == 'Disease') {
            markerColor = BitmapDescriptor.hueRed;
          } else {
            markerColor = item['iot_status'] == 'Confirmed' 
                ? BitmapDescriptor.hueBlue 
                : BitmapDescriptor.hueOrange;
          }

          return Marker(
            markerId: MarkerId(item['id'].toString()),
            position: LatLng(lat, lng),
            icon: BitmapDescriptor.defaultMarkerWithHue(markerColor),
            infoWindow: InfoWindow(
              title: item['prediction'],
              snippet: "Confidence: ${(item['confidence'] * 100).toStringAsFixed(1)}%",
            ),
          );
        }).toSet();

        setState(() {
          _markers = tempMarkers;
          _isLoading = false;
        });
      }
    } catch (e) {
      print("❌ Map Loading Error: $e");
      setState(() { _isLoading = false; });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Rice Field Analysis Map"), 
        backgroundColor: Colors.green
      ),
      body: _isLoading 
          ? const Center(child: CircularProgressIndicator()) // Load වෙනකම් පෙන්වයි
          : GoogleMap(
              initialCameraPosition: const CameraPosition(
                target: LatLng(7.8731, 80.7718), // ශ්‍රී ලංකාවේ මැද ලක්ෂ්‍යයට සකස් කළා
                zoom: 7,
              ),
              markers: _markers,
              myLocationEnabled: true,
              mapType: MapType.hybrid, // කුඹුරු පෙන්වීමට වඩාත් සුදුසුයි
            ),
    );
  }
}