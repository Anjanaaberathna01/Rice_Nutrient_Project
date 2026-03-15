import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'map_screen.dart'; // දැන් මෙය පාවිච්චි වන නිසා Warning එකක් එන්නේ නැත
import 'package:geolocator/geolocator.dart';
import 'package:image/image.dart' as img;

void main() {
  runApp(const RiceApp());
}

class RiceApp extends StatelessWidget {
  const RiceApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(primarySwatch: Colors.green),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  File? _selectedImage;
  String _result = "No Analysis Yet";
  bool _isLoading = false;
  double? _lat;
  double? _lng;

  Future<Position> _determinePosition() async {
    bool serviceEnabled;
    LocationPermission permission;

    serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) return Future.error('Location services are disabled.');

    permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) return Future.error('Location permissions are denied');
    }
    
    if (permission == LocationPermission.deniedForever) return Future.error('Location permissions are permanently denied.');

    return await Geolocator.getCurrentPosition(
      locationSettings: const LocationSettings(accuracy: LocationAccuracy.high)
    );
  }

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final XFile? image = await picker.pickImage(
      source: ImageSource.camera,
      imageQuality: 100, 
    );

    if (image != null) {
      setState(() {
        _selectedImage = File(image.path);
        _result = "Captured. Review and Analyze.";
        _lat = null;
        _lng = null;
      });
    }
  }

  Future<void> _uploadImage() async {
    if (_selectedImage == null) return;

    setState(() { 
      _isLoading = true; 
      _result = "Processing Image & Analyzing..."; 
    });

    try {
      // --- පින්තූරය සැබෑ ලෙසම Crop කිරීමේ ක්‍රියාවලිය ---
      final bytes = await _selectedImage!.readAsBytes();
      img.Image? originalImage = img.decodeImage(bytes);

      if (originalImage == null) throw Exception("Could not decode image");

      // පටු සිරස් කොටසක් මැදින් කපා ගමු (Width 40% ට අඩු කරනවා)
      int cropWidth = (originalImage.width * 0.4).toInt(); 
      int cropHeight = originalImage.height; 

      img.Image croppedImage = img.copyCrop(
        originalImage, 
        x: (originalImage.width - cropWidth) ~/ 2,
        y: 0, 
        width: cropWidth, 
        height: cropHeight
      );

      final tempDir = Directory.systemTemp;
      final croppedFile = File('${tempDir.path}/temp_crop.jpg');
      await croppedFile.writeAsBytes(img.encodeJpg(croppedImage));
      // ---------------------------------------

      Position position = await _determinePosition();
      _lat = position.latitude;
      _lng = position.longitude;
      
      var uri = Uri.parse('http://192.168.8.163:5000/predict'); 
      var request = http.MultipartRequest('POST', uri);
      
      request.files.add(await http.MultipartFile.fromPath('image', croppedFile.path));
      request.fields['lat'] = _lat.toString();
      request.fields['lng'] = _lng.toString();

      var response = await request.send();
      var responseData = await response.stream.bytesToString();
      var decodedData = json.decode(responseData);

      setState(() {
        if (decodedData['is_rice'] == true) {
          _result = "Status: ${decodedData['prediction']}\n"
                    "Confidence: ${(decodedData['confidence'] * 100).toStringAsFixed(2)}%\n"
                    "Solution: ${decodedData['solution']}";
        } else {
          _lat = null; _lng = null;
          _result = "Message: ${decodedData['message']}";
        }
      });
    } catch (e) {
      setState(() { _result = "Error: $e"; });
    } finally {
      setState(() { _isLoading = false; });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Rice Leaf Analyzer"),
        backgroundColor: Colors.green[700],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          child: Column(
            children: [
              const SizedBox(height: 15),
              
              Center(
                child: Container(
                  width: 200, 
                  height: 400, 
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.green, width: 3),
                    borderRadius: BorderRadius.circular(15),
                    color: Colors.black12,
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(12),
                    child: _selectedImage == null
                        ? const Center(child: Icon(Icons.camera_alt, size: 50, color: Colors.grey))
                        : Image.file(_selectedImage!, fit: BoxFit.cover),
                  ),
                ),
              ),
              
              const SizedBox(height: 10),

              const Padding(
                padding: EdgeInsets.symmetric(horizontal: 40),
                child: Column(
                  children: [
                    Row(children: [Icon(Icons.check_circle, color: Colors.green, size: 14), SizedBox(width: 8), Text("දුරකථනය සිරස් අතට අල්ලන්න", style: TextStyle(fontSize: 12))]),
                    Row(children: [Icon(Icons.check_circle, color: Colors.green, size: 14), SizedBox(width: 8), Text("පත්‍රය කොටුව මැදට ගන්න", style: TextStyle(fontSize: 12))]),
                  ],
                ),
              ),

              const SizedBox(height: 15),
              
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20),
                child: Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.grey[200],
                    borderRadius: BorderRadius.circular(15),
                  ),
                  child: Text(
                    _result,
                    textAlign: TextAlign.center,
                    style: const TextStyle(fontSize: 13, fontWeight: FontWeight.bold),
                  ),
                ),
              ),
              
              const SizedBox(height: 15),
              
              _isLoading 
                ? const CircularProgressIndicator() 
                : Column(
                    children: [
                      ElevatedButton.icon(
                        onPressed: _pickImage,
                        icon: const Icon(Icons.camera_alt),
                        label: const Text("Capture Leaf Photo"),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.blueAccent,
                          foregroundColor: Colors.white,
                          minimumSize: const Size(240, 45),
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(30))
                        ),
                      ),
                      const SizedBox(height: 12),
                      if (_selectedImage != null)
                        ElevatedButton.icon(
                          onPressed: _uploadImage,
                          icon: const Icon(Icons.search),
                          label: const Text("Analyze Leaf"),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.green[600], 
                            foregroundColor: Colors.white,
                            minimumSize: const Size(240, 45),
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(30))
                          ),
                        ),
                      
                      // --- Map Button එක ආපසු එකතු කළා ---
                      if (_lat != null && _lng != null)
                        Padding(
                          padding: const EdgeInsets.only(top: 12),
                          child: ElevatedButton.icon(
                            onPressed: () {
                              Navigator.push(
                                context,
                                MaterialPageRoute(
                                  builder: (context) => MapScreen(
                                    lat: _lat!, lng: _lng!, status: _result,
                                  ),
                                ),
                              );
                            },
                            icon: const Icon(Icons.map),
                            label: const Text("View on Map"),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.orange[700], 
                              foregroundColor: Colors.white,
                              minimumSize: const Size(240, 45),
                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(30))
                            ),
                          ),
                        ),
                    ],
                  ),
              const SizedBox(height: 30),
            ],
          ),
        ),
      ),
    );
  }
}