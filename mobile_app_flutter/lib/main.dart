import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:geolocator/geolocator.dart';
import 'package:image/image.dart' as img;
import 'map_screen.dart';

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

  // 💡 ත්‍රිත්වකරණයට අවශ්‍ය Global Variables
  String? _currentPredictionId;
  String? _rawPredictionClass;
  String? _rawCategory;

  final List<String> _dsDivisions = ['anuradhapura'];
  final Map<String, List<String>> _gnDivisions = {
    'anuradhapura': ['srwasthipura'], 
  };

  Future<Position> _determinePosition() async {
    bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) return Future.error('Location services are disabled.');

    LocationPermission permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) return Future.error('Location permissions are denied');
    }
    return await Geolocator.getCurrentPosition(locationSettings: const LocationSettings(accuracy: LocationAccuracy.high));
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
        _currentPredictionId = null;
        _rawPredictionClass = null;
        _rawCategory = null;
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
      
      var uri = Uri.parse('http://192.168.8.163:5000/predict'); 
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
          
          _currentPredictionId = decodedData['prediction_id']?.toString();
          _rawPredictionClass = decodedData['prediction'];
          _rawCategory = decodedData['category'];
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

  void _showLocationSearchDialog() {
    final TextEditingController dsController = TextEditingController();
    final TextEditingController gnController = TextEditingController();
    String activeDS = '';

    showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text("ප්‍රදේශය ඇතුළත් කරන්න (Type to Search)"),
          content: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Autocomplete<String>(
                  optionsBuilder: (TextEditingValue textEditingValue) {
                    if (textEditingValue.text.isEmpty) return const Iterable<String>.empty();
                    return _dsDivisions.where((String option) => option.contains(textEditingValue.text.toLowerCase()));
                  },
                  onSelected: (String selection) {
                    activeDS = selection;
                    dsController.text = selection;
                  },
                  fieldViewBuilder: (context, controller, focusNode, onFieldSubmitted) {
                    return TextField(
                      controller: controller,
                      focusNode: focusNode,
                      decoration: const InputDecoration(labelText: "ප්‍රාදේශීය ලේකම් කොට්ඨාශය", prefixIcon: Icon(Icons.search)),
                    );
                  },
                ),
                const SizedBox(height: 15),
                Autocomplete<String>(
                  optionsBuilder: (TextEditingValue textEditingValue) {
                    if (textEditingValue.text.isEmpty || activeDS.isEmpty) return const Iterable<String>.empty();
                    return _gnDivisions[activeDS]!.where((String option) => option.contains(textEditingValue.text.toLowerCase()));
                  },
                  onSelected: (String selection) {
                    gnController.text = selection;
                  },
                  fieldViewBuilder: (context, controller, focusNode, onFieldSubmitted) {
                    return TextField(
                      controller: controller,
                      focusNode: focusNode,
                      decoration: const InputDecoration(labelText: "ග්‍රාම නිලධාරී වසම", prefixIcon: Icon(Icons.location_city)),
                    );
                  },
                ),
              ],
            ),
          ),
          actions: [
            TextButton(onPressed: () => Navigator.pop(context), child: const Text("Cancel")),
            ElevatedButton(
              onPressed: () {
                if (gnController.text.isNotEmpty && _currentPredictionId != null) {
                  Navigator.pop(context);
                  _fetchFertilizerRecommendation(
                    _currentPredictionId!,
                    gnController.text.trim(),
                    _rawPredictionClass!,
                    _rawCategory!,
                  );
                }
              },
              child: const Text("Validate & View Advice"),
            )
          ],
        );
      },
    );
  }

  Future<void> _fetchFertilizerRecommendation(String predId, String gn, String predClass, String cat) async {
    setState(() { _isLoading = true; });
    try {
      var uri = Uri.parse('http://192.168.8.163:5000/validate_and_recommend_fertilizer');
      var response = await http.post(
        uri,
        headers: {"Content-Type": "application/json"},
        body: json.encode({
          "prediction_id": predId,
          "gn_division": gn,
          "prediction": predClass,
          "category": cat
        }),
      );

      if (response.statusCode == 200 && mounted) {
        var resData = json.decode(response.body);
        setState(() { _isLoading = false; });
        _showFertilizerResultDialog(resData['recommendation'] ?? 'දත්ත විශ්ලේෂණය අවසන්.');
      } else {
        setState(() { _isLoading = false; });
      }
    } catch (e) {
      if (mounted) {
        setState(() { _isLoading = false; });
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Error: $e")));
      }
    }
  }

  void _showFertilizerResultDialog(String recommendationMessage) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Row(
          children: [
            Icon(Icons.gavel_rounded, color: Colors.green),
            SizedBox(width: 10),
            Text("විද්‍යාත්මක ත්‍රිත්වකරණය"),
          ],
        ),
        content: SingleChildScrollView(
          child: Text(recommendationMessage, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600, height: 1.5, color: Colors.black87)),
        ),
        actions: [TextButton(onPressed: () => Navigator.pop(context), child: const Text("Close", style: TextStyle(fontWeight: FontWeight.bold)))],
      ),
    );
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
                      decoration: BoxDecoration(border: Border.all(color: Colors.green.withValues(alpha: 0.5)), borderRadius: BorderRadius.circular(8)),
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
                    // 💡 Healthy නොවන ඕනෑම අවස්ථාවක (ඌණතා සහ රෝග දෙකටම) බටන් එක පෙන්වයි bn
                    if (_currentPredictionId != null && _rawPredictionClass != 'Healthy')
                      ElevatedButton.icon(
                        onPressed: _showLocationSearchDialog,
                        icon: const Icon(Icons.spoke_outlined),
                        label: const Text("Get Fertilizer Advice"),
                        style: ElevatedButton.styleFrom(
                          minimumSize: const Size(260, 50), 
                          backgroundColor: Colors.purple[700], 
                          foregroundColor: Colors.white, 
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(30))
                        ),
                      ),
                    const SizedBox(height: 12),
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
                  ],
                ),
            const SizedBox(height: 30),
          ],
        ),
      ),
    );
  }
}