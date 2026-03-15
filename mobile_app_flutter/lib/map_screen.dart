import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';

class MapScreen extends StatelessWidget {
  final double lat;
  final double lng;
  final String status;

  const MapScreen({super.key, required this.lat, required this.lng, required this.status});

  @override
  Widget build(BuildContext context) {
    if (lat == 0.0 || lng == 0.0) {
    return const Scaffold(body: Center(child: Text("Invalid Location Data")));
    }
    return Scaffold(
      appBar: AppBar(
        title: const Text("Location Map"),
        backgroundColor: Colors.green,
      ),
      body: GoogleMap(
        initialCameraPosition: CameraPosition(
          target: LatLng(lat, lng),
          zoom: 16,
        ),
        markers: {
          Marker(
            markerId: const MarkerId('paddy_loc'),
            position: LatLng(lat, lng),
            infoWindow: InfoWindow(
              title: "Analysis Result",
              snippet: status,
            ),
            icon: BitmapDescriptor.defaultMarkerWithHue(BitmapDescriptor.hueOrange), // පර්යේෂණයේ පරිදි Orange Marker එක
          ),
        },
      ),
    );
  }
}