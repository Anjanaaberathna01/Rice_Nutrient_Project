import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:geolocator/geolocator.dart';
import 'package:image/image.dart' as img;
import 'map_screen.dart'; // Map Screen එක import කරගන්න

void main() {
  runApp(const RiceApp());
}

class RiceApp extends StatelessWidget {
  const RiceApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(primarySwatch: Colors.green, useMaterial3: true),
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
  String _result = "තවමත් විශ්ලේෂණය කර නැත";
  String? _heatmapBase64; 
  bool _isLoading = false;
  double? _lat;
  double? _lng;

  Future<Position> _determinePosition() async {
    bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) return Future.error('Location services are disabled.');

    LocationPermission permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) return Future.error('Location permissions are denied');
    }
    return await Geolocator.getCurrentPosition(desiredAccuracy: LocationAccuracy.high);
  }

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final XFile? image = await picker.pickImage(source: ImageSource.camera, imageQuality: 100);

    if (image != null) {
      setState(() {
        _selectedImage = File(image.path);
        _result = "පින්තූරය ලබාගත්තා. දැන් Analyze කරන්න.";
        _heatmapBase64 = null;
        _lat = null; _lng = null;
      });
    }
  }

  Future<void> _uploadImage() async {
    if (_selectedImage == null) return;

    setState(() { 
      _isLoading = true; 
      _result = "පින්තූරය පිරිසැකසුම් කරමින් විශ්ලේෂණය කරයි..."; 
    });

    try {
      final bytes = await _selectedImage!.readAsBytes();
      img.Image? originalImage = img.decodeImage(bytes);
      if (originalImage == null) throw Exception("Could not decode image");

      int cropWidth = (originalImage.width * 0.4).toInt(); 
      img.Image croppedImage = img.copyCrop(
        originalImage, 
        x: (originalImage.width - cropWidth) ~/ 2, y: 0, 
        width: cropWidth, height: originalImage.height
      );

      final tempDir = Directory.systemTemp;
      final croppedFile = File('${tempDir.path}/temp_crop.jpg');
      await croppedFile.writeAsBytes(img.encodeJpg(croppedImage));

      Position position = await _determinePosition();
      _lat = position.latitude;
      _lng = position.longitude;
      
      var uri = Uri.parse('http://192.168.8.164:5000/predict'); 
      var request = http.MultipartRequest('POST', uri);
      
      request.files.add(await http.MultipartFile.fromPath('image', croppedFile.path));
      request.fields['lat'] = _lat.toString();
      request.fields['lng'] = _lng.toString();

      var response = await request.send();
      var responseData = await response.stream.bytesToString();
      var decodedData = json.decode(responseData);

      setState(() {
        if (decodedData['status'] == 'success') {
          double dis = (decodedData['disease_score'] ?? 0.0) * 100;
          double def = (decodedData['deficiency_score'] ?? 0.0) * 100;
          double diff = (dis - def).abs();

          String statusTag = diff < 15 
              ? "⚠️ මිශ්‍ර ලක්ෂණ පවතින බවට සැක කෙරේ\n" 
              : "✅ සාර්ථකව හඳුනා ගන්නා ලදී\n";

          _result = "$statusTag"
                    "----------------------------------\n"
                    "ප්‍රධාන නිගමනය: ${decodedData['prediction']}\n"
                    "විශ්වාසනීයත්වය: ${(decodedData['confidence'] * 100).toStringAsFixed(2)}%\n"
                    "----------------------------------\n"
                    "රෝග ලක්ෂණ: ${dis.toStringAsFixed(0)}%\n"
                    "ඌණතා ලක්ෂණ: ${def.toStringAsFixed(0)}%\n"
                    "----------------------------------\n"
                    "සටහන: ${decodedData['message']}";
          
          _heatmapBase64 = decodedData['heatmap'];
          if (_heatmapBase64 != null) _showHeatmapDialog();
          
        } else {
          _result = "දෝෂයකි: ${decodedData['message']}";
        }
      });
    } catch (e) {
      setState(() { _result = "සම්බන්ධතා දෝෂයකි: $e"; });
    } finally {
      setState(() { _isLoading = false; });
    }
  }

  void _showHeatmapDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text("Ensemble Attention Map", style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
        content: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text("මොඩලය පත්‍රයේ අවධානය යොමු කළ ස්ථාන මෙහි වර්ණ ගැන්වී පෙන්වයි.", style: TextStyle(fontSize: 12), textAlign: TextAlign.center),
              const SizedBox(height: 15),
              _heatmapBase64 != null
                  ? Container(
                      constraints: BoxConstraints(
                        maxHeight: MediaQuery.of(context).size.height * 0.5,
                        maxWidth: MediaQuery.of(context).size.width * 0.8,
                      ),
                      decoration: BoxDecoration(border: Border.all(color: Colors.green.withOpacity(0.5)), borderRadius: BorderRadius.circular(8)),
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(8),
                        child: Image.memory(base64Decode(_heatmapBase64!), fit: BoxFit.contain),
                      ),
                    )
                  : const Text("Heatmap ලබා ගත නොහැක"),
            ],
          ),
        ),
        actions: [TextButton(onPressed: () => Navigator.pop(context), child: const Text("Close"))],
      ),
    );
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Rice Nutrient & Disease Analyzer"),
        backgroundColor: Colors.green[700],
        foregroundColor: Colors.white,
        actions: [
          IconButton(
            onPressed: () {
              Navigator.push(context, MaterialPageRoute(builder: (context) => const MapScreen()));
            }, 
            icon: const Icon(Icons.map_rounded)
          )
        ],
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            const SizedBox(height: 20),
            Center(
              child: Container(
                width: 220, height: 400,
                decoration: BoxDecoration(border: Border.all(color: Colors.green, width: 3), borderRadius: BorderRadius.circular(15)),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: _selectedImage == null
                      ? const Icon(Icons.center_focus_weak, size: 50, color: Colors.grey)
                      : Image.file(_selectedImage!, fit: BoxFit.cover),
                ),
              ),
            ),
            const SizedBox(height: 20),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Card(
                elevation: 4,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
                child: Padding(
                  padding: const EdgeInsets.all(20),
                  child: Text(_result, textAlign: TextAlign.left, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600, height: 1.6)),
                ),
              ),
            ),
            const SizedBox(height: 20),
            _isLoading 
              ? const CircularProgressIndicator(color: Colors.green)
              : Column(
                  children: [
                    ElevatedButton.icon(
                      onPressed: _pickImage,
                      icon: const Icon(Icons.camera_alt),
                      label: const Text("Capture Leaf"),
                      style: ElevatedButton.styleFrom(minimumSize: const Size(260, 50), backgroundColor: Colors.blueAccent, foregroundColor: Colors.white, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(30))),
                    ),
                    const SizedBox(height: 12),
                    if (_selectedImage != null)
                      ElevatedButton.icon(
                        onPressed: _uploadImage,
                        icon: const Icon(Icons.analytics_outlined),
                        label: const Text("Analyze Now"),
                        style: ElevatedButton.styleFrom(minimumSize: const Size(260, 50), backgroundColor: Colors.green[600], foregroundColor: Colors.white, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(30))),
                      ),
                    const SizedBox(height: 12),
                    // --- Map එක පෙන්වන අලුත් Button එක ---
                    ElevatedButton.icon(
                      onPressed: () {
                        Navigator.push(context, MaterialPageRoute(builder: (context) => const MapScreen()));
                      },
                      icon: const Icon(Icons.map_outlined),
                      label: const Text("View Analysis Map"),
                      style: ElevatedButton.styleFrom(
                        minimumSize: const Size(260, 50), 
                        backgroundColor: Colors.orange[800], 
                        foregroundColor: Colors.white, 
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(30))
                      ),
                    ),
                    if (_heatmapBase64 != null)
                       Padding(
                         padding: const EdgeInsets.only(top: 10),
                         child: TextButton.icon(onPressed: _showHeatmapDialog, icon: const Icon(Icons.remove_red_eye_outlined), label: const Text("View Heatmap")),
                       ),
                  ],
                ),
            const SizedBox(height: 30),
          ],
        ),
      ),
    );
  }
}