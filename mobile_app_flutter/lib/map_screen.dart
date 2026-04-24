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

  Future<void> _fetchLatestLocation() async {
    try {
      // ඔබගේ වත්මන් Backend IP එක පාවිච්චි කරන්න
      final response = await http.get(Uri.parse('http://192.168.8.164:5000/get_latest_location'));

      if (response.statusCode == 200) {
        final List<dynamic> data = json.decode(response.body);
        
        if (data.isNotEmpty) {
          final item = data[0];
          double lat = double.tryParse(item['latitude'].toString()) ?? 0.0;
          double lng = double.tryParse(item['longitude'].toString()) ?? 0.0;
          
          // Category අනුව වර්ණය තීරණය කිරීම
          double markerColor = item['category'] == 'Disease' 
              ? BitmapDescriptor.hueRed 
              : BitmapDescriptor.hueOrange;

          setState(() {
            _markers = {
              Marker(
                markerId: MarkerId(item['id'].toString()),
                position: LatLng(lat, lng),
                icon: BitmapDescriptor.defaultMarkerWithHue(markerColor),
                infoWindow: InfoWindow(
                  title: "නිගමනය: ${item['prediction']}",
                  snippet: "විශ්වාසනීයත්වය: ${(item['confidence'] * 100).toStringAsFixed(1)}%",
                ),
              )
            };
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
                // අන්තිම ප්‍රතිඵලය ඇති ස්ථානයටම සිතියම Zoom කරවයි
                target: _markers.isNotEmpty 
                    ? _markers.first.position 
                    : const LatLng(7.8731, 80.7718),
                zoom: 16, // මීටර් 10-15 පරාසය බැලීමට සුදුසු Zoom අගයකි
              ),
              markers: _markers,
              mapType: MapType.hybrid, // කුඹුරු සහ ගස් හොඳින් පෙනේ
              myLocationEnabled: true,
            ),
    );
  }
}