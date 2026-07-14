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

  GoogleMapController? _mapController;

  @override
  void initState() {
    super.initState();

    _fetchLatestLocation();
  }

  double _getMarkerHue(String? color) {
    switch (color) {
      case "GREEN":
        return BitmapDescriptor.hueGreen;

      case "RED":
        return BitmapDescriptor.hueRed;

      default:
        return BitmapDescriptor.hueOrange;
    }
  }

  Future<void> _fetchLatestLocation() async {
    try {
      final response = await http.get(
        Uri.parse('http://192.168.8.163:5000/get_latest_location'),
      );

      if (response.statusCode == 200) {
        List<dynamic> data = json.decode(response.body);

        Set<Marker> newMarkers = {};

        LatLng? latestPosition;

        for (var item in data) {
          if (item['latitude'] == null || item['longitude'] == null) {
            continue;
          }

          double lat = double.parse(item['latitude'].toString());

          double lng = double.parse(item['longitude'].toString());

          LatLng position = LatLng(lat, lng);

          latestPosition = position;

          newMarkers.add(
            Marker(
              markerId: MarkerId(item['id'].toString()),

              position: position,

              icon: BitmapDescriptor.defaultMarkerWithHue(
                _getMarkerHue(item['marker_color']),
              ),

              infoWindow: InfoWindow(
                title: item['prediction'] ?? "Unknown",

                snippet:
                    "Status : ${item['status']}\n"
                    "Category : ${item['category']}\n"
                    "Confidence : "
                    "${((double.tryParse(item['confidence'].toString()) ?? 0) * 100).toStringAsFixed(1)}%",
              ),
            ),
          );
        }

        setState(() {
          _markers = newMarkers;

          _isLoading = false;
        });

        if (latestPosition != null && _mapController != null) {
          _mapController!.animateCamera(
            CameraUpdate.newLatLngZoom(latestPosition, 17),
          );
        }
      }
    } catch (e) {
      debugPrint("MAP ERROR : $e");

      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Analysis Map"),

        backgroundColor: Colors.green[700],

        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),

            onPressed: _fetchLatestLocation,
          ),
        ],
      ),

      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : GoogleMap(
              onMapCreated: (controller) {
                _mapController = controller;
              },

              initialCameraPosition: const CameraPosition(
                target: LatLng(7.8731, 80.7718),

                zoom: 8,
              ),

              markers: _markers,

              mapType: MapType.hybrid,
            ),
    );
  }
}
