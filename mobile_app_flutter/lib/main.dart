import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:typed_data';
import 'package:gal/gal.dart';
import 'package:permission_handler/permission_handler.dart';
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

  String? _currentPredictionId;
  String? _rawPredictionClass;
  String? _rawCategory;

  // dynamic variables for DS and GN divisions
  String? _selectedDS;
  String? _selectedGN;

  List<String> _dsDivisions = [];
  Map<String, List<String>> _gnDivisions = {};
  bool _isDivisionsLoading = true;

  @override
  void initState() {
    super.initState();
    _fetchDivisionsFromBackend();
  }

  //backend eken divisions ganna function eka
  Future<void> _fetchDivisionsFromBackend() async {
    try {
      var response = await http.get(
        Uri.parse('http://192.168.1.103:5000/api/get_divisions'),
      );

      if (response.statusCode == 200) {
        var data = json.decode(response.body);
        if (data['status'] == 'success') {
          setState(() {
            _dsDivisions = List<String>.from(data['ds_divisions']);
            
            var rawGnMap = data['gn_divisions'] as Map<String, dynamic>;
            _gnDivisions = rawGnMap.map((key, value) {
              return MapEntry(key, List<String>.from(value));
            });

            if (_dsDivisions.isNotEmpty) {
              _selectedDS = _dsDivisions.first;
              if (_gnDivisions[_selectedDS!] != null && _gnDivisions[_selectedDS!]!.isNotEmpty) {
                _selectedGN = _gnDivisions[_selectedDS!]!.first;
              }
            }
            _isDivisionsLoading = false;
          });
        }
      }
    } catch (e) {
      print("Error fetching divisions: $e");
      setState(() {
        _isDivisionsLoading = false;
      });
    }
  }

  Future<Position> _determinePosition() async {
    bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) return Future.error('Location services are disabled.');

    LocationPermission permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied){
        return Future.error('Location permissions are denied');
      }
    }
    return await Geolocator.getCurrentPosition(
      locationSettings: const LocationSettings(accuracy: LocationAccuracy.high),
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
        _result = "පින්තූරය ලබාගත්තා. වසම තෝරා Analyze Now ඔබන්න.";
        _heatmapBase64 = null;
        _lat = null;
        _lng = null;
        _currentPredictionId = null;
        _rawPredictionClass = null;
        _rawCategory = null;
      });
    }
  }

  // In click Analyze Now (Predict + Soil/Weather Validation)
  Future<void> _uploadImageAndValidate() async {
    if (_selectedImage == null) return;

    if (_selectedDS == null || _selectedGN == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("කරුණාකර ප්‍රාදේශීය ලේකම් සහ ග්‍රාම නිලධාරී වසම තෝරන්න.")),
      );
      return;
    }

    setState(() {
      _isLoading = true;
      _result = "පින්තූරය විශ්ලේෂණය කරමින් සහ පස්/කාලගුණ දත්ත සසඳමින් පවතී...";
    });

    try {
      final bytes = await _selectedImage!.readAsBytes();
      img.Image? originalImage = img.decodeImage(bytes);
      if (originalImage == null) throw Exception("Could not decode image");

      int cropWidth = (originalImage.width * 0.4).toInt();
      img.Image croppedImage = img.copyCrop(
        originalImage,
        x: (originalImage.width - cropWidth) ~/ 2,
        y: 0,
        width: cropWidth,
        height: originalImage.height,
      );

      final tempDir = Directory.systemTemp;
      final croppedFile = File('${tempDir.path}/temp_crop.jpg');
      await croppedFile.writeAsBytes(img.encodeJpg(croppedImage));

      Position position = await _determinePosition();
      _lat = position.latitude;
      _lng = position.longitude;

      // 1. Predict API Call
      var uri = Uri.parse('http://192.168.1.103:5000/predict');
      var request = http.MultipartRequest('POST', uri);

      request.files.add(
        await http.MultipartFile.fromPath('image', croppedFile.path),
      );
      request.fields['lat'] = _lat.toString();
      request.fields['lng'] = _lng.toString();

      var response = await request.send();
      var responseData = await response.stream.bytesToString();
      var decodedData = json.decode(responseData);

      if (decodedData['status'] == 'success') {
        _currentPredictionId = decodedData['prediction_id']?.toString();
        _rawPredictionClass = decodedData['prediction'];
        _rawCategory = decodedData['category'];
        _heatmapBase64 = decodedData['heatmap'];

        double dis = (decodedData['disease_score'] ?? 0.0) * 100;
        double def = (decodedData['deficiency_score'] ?? 0.0) * 100;

        // 2. AI ප්‍රතිඵලය ලැබුණු වහාම ස්වයංක්‍රීයව Validation API Call එක යැවීම
        if (_currentPredictionId != null) {
          await _fetchSoilAndWeatherPrediction(
            _currentPredictionId!,
            _selectedDS!,
            _selectedGN!,
            _rawPredictionClass!,
            _rawCategory!,
            _lat,
            _lng,
            dis,
            def,
            decodedData['confidence'],
          );
        }

        if (_heatmapBase64 != null) _showHeatmapDialog();
      } else {
        setState(() {
          _isLoading = false;
          _result = "දෝෂයකි: ${decodedData['message']}";
        });
      }
    } catch (e) {
      setState(() {
        _isLoading = false;
        _result = "සම්බන්ධතා දෝෂයකි: $e";
      });
    }
  }

  // 💡 /validate_and_recommend_fertilizer වෙත API Call එක යැවීම සහ ප්‍රතිඵලය පෙන්වීම
  Future<void> _fetchSoilAndWeatherPrediction(
    String predId,
    String ds,
    String gn,
    String predClass,
    String cat,
    double? lat,
    double? lng,
    double disScore,
    double defScore,
    double confidence,
  ) async {
    try {
      var uri = Uri.parse(
        'http://192.168.1.103:5000/validate_and_recommend_fertilizer',
      );

      var response = await http.post(
        uri,
        headers: {"Content-Type": "application/json"},
        body: json.encode({
          "prediction_id": predId,
          "ds_division": ds,
          "gn_division": gn,
          "prediction": predClass,
          "category": cat,
          "lat": lat,
          "lng": lng,
        }),
      );

      if (response.statusCode == 200 && mounted) {
        var resData = json.decode(response.body);
        String recommendation = resData['recommendation'] ?? "විස්තර ලබා ගත නොහැක";

        setState(() {
          _isLoading = false;
          _result = " සාර්ථකව හඳුනා ගන්නා ලදී\n"
              "----------------------------------\n"
              "ප්‍රධාන නිගමනය: $predClass\n"
              "විශ්වාසනීයත්වය: ${(confidence * 100).toStringAsFixed(2)}%\n"
              "----------------------------------\n"
              "රෝග ලක්ෂණ: ${disScore.toStringAsFixed(0)}%\n"
              "ඌණතා ලක්ෂණ: ${defScore.toStringAsFixed(0)}%\n"
              "----------------------------------\n"
              "පස් හා කාලගුණ විද්‍යාත්මක පුරෝකථනය:\n"
              "$recommendation";
        });
      } else {
        setState(() {
          _isLoading = false;
        });
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Server Error : ${response.statusCode}")),
        );
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Connection Error : $e")),
        );
      }
    }
  }

  Future<void> _downloadHeatmap() async {
    if (_heatmapBase64 == null) return;
    
    var status = await Permission.storage.request();
    if (!status.isGranted) {
      status = await Permission.photos.request();
    }
    
    if (status.isGranted) {
      try {
        Uint8List imageBytes = base64Decode(_heatmapBase64!);
        await Gal.putImageBytes(imageBytes);

        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Heatmap Downloaded Successfully! ✅'), backgroundColor: Colors.green),
          );
        }
      } catch (e) {
        print("Error saving image: $e");
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Failed to save image. ❌'), backgroundColor: Colors.red),
          );
        }
      }
    } else {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Storage Permission Denied! ⚠️')),
        );
      }
    }
  }

  void _showHeatmapDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text(
          "Ensemble Attention Map",
          style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
        ),
        content: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text(
                "මොඩලය පත්‍රයේ අවධානය යොමු කළ ස්ථාන මෙහි වර්ණ ගැන්වී පෙන්වයි.",
                style: TextStyle(fontSize: 12),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 15),
              _heatmapBase64 != null
                  ? Container(
                      constraints: BoxConstraints(
                        maxHeight: MediaQuery.of(context).size.height * 0.5,
                        maxWidth: MediaQuery.of(context).size.width * 0.8,
                      ),
                      decoration: BoxDecoration(
                        border: Border.all(
                          color: Colors.green.withValues(alpha: 0.5),
                        ),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(8),
                        child: Image.memory(
                          base64Decode(_heatmapBase64!),
                          fit: BoxFit.contain,
                        ),
                      ),
                    )
                  : const Text("Heatmap ලබා ගත නොහැක"),
            ],
          ),
        ),
        actions: [
          TextButton.icon(
            onPressed: () {
              _downloadHeatmap();
              Navigator.pop(context);
            },
            icon: const Icon(Icons.download),
            label: const Text("Download"),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text("Close"),
          ),
        ],
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
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => const MapScreen()),
              );
            },
            icon: const Icon(Icons.map_rounded),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            // 💡 වසම් තෝරාගැනීමේ කොටස (Autocomplete TextFields)
            Card(
              elevation: 3,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              child: Padding(
                padding: const EdgeInsets.all(12.0),
                child: _isDivisionsLoading
                    ? const Center(child: Padding(
                        padding: EdgeInsets.all(8.0),
                        child: CircularProgressIndicator(),
                      ))
                    : Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            "වගා කරන ප්‍රදේශය ඇතුළත් කරන්න:",
                            style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14),
                          ),
                          const SizedBox(height: 10),
                          
                          // 1. DS Division Autocomplete
                          Autocomplete<String>(
                            optionsBuilder: (TextEditingValue textEditingValue) {
                              if (textEditingValue.text.isEmpty) {
                                return const Iterable<String>.empty();
                              }
                              return _dsDivisions.where((String ds) =>
                                  ds.toLowerCase().contains(textEditingValue.text.toLowerCase()));
                            },
                            onSelected: (String selection) {
                              setState(() {
                                _selectedDS = selection;
                                _selectedGN = _gnDivisions[_selectedDS!]?.first;
                              });
                            },
                            fieldViewBuilder: (context, controller, focusNode, onFieldSubmitted) {
                              if (_selectedDS != null && controller.text.isEmpty) {
                                controller.text = _selectedDS!;
                              }
                              return TextField(
                                controller: controller,
                                focusNode: focusNode,
                                decoration: const InputDecoration(
                                  labelText: "ප්‍රාදේශීය ලේකම් කොට්ඨාශය (DS) ටයිප් කරන්න",
                                  border: OutlineInputBorder(),
                                  isDense: true,
                                  prefixIcon: Icon(Icons.search),
                                ),
                              );
                            },
                          ),
                          const SizedBox(height: 10),

                          // 2. GN Division Autocomplete
                          Autocomplete<String>(
                            optionsBuilder: (TextEditingValue textEditingValue) {
                              if (textEditingValue.text.isEmpty || _selectedDS == null) {
                                return const Iterable<String>.empty();
                              }
                              List<String> availableGns = _gnDivisions[_selectedDS!] ?? [];
                              return availableGns.where((String gn) =>
                                  gn.toLowerCase().contains(textEditingValue.text.toLowerCase()));
                            },
                            onSelected: (String selection) {
                              setState(() {
                                _selectedGN = selection;
                              });
                            },
                            fieldViewBuilder: (context, controller, focusNode, onFieldSubmitted) {
                              if (_selectedGN != null && controller.text.isEmpty) {
                                controller.text = _selectedGN!;
                              }
                              return TextField(
                                controller: controller,
                                focusNode: focusNode,
                                decoration: const InputDecoration(
                                  labelText: "ග්‍රාම නිලධාරී වසම (GN) ටයිප් කරන්න",
                                  border: OutlineInputBorder(),
                                  isDense: true,
                                  prefixIcon: Icon(Icons.location_city),
                                ),
                              );
                            },
                          ),
                        ],
                      ),
              ),
            ),
            const SizedBox(height: 15),
            Center(
              child: Container(
                width: 200,
                height: 320,
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.green, width: 3),
                  borderRadius: BorderRadius.circular(15),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: _selectedImage == null
                      ? const Icon(
                          Icons.center_focus_weak,
                          size: 50,
                          color: Colors.grey,
                        )
                      : Image.file(_selectedImage!, fit: BoxFit.cover),
                ),
              ),
            ),
            const SizedBox(height: 15),
            Card(
              elevation: 4,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(15),
              ),
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Text(
                  _result,
                  textAlign: TextAlign.left,
                  style: const TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                    height: 1.6,
                  ),
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
                        style: ElevatedButton.styleFrom(
                          minimumSize: const Size(260, 50),
                          backgroundColor: Colors.blueAccent,
                          foregroundColor: Colors.white,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(30),
                          ),
                        ),
                      ),
                      const SizedBox(height: 12),
                      if (_selectedImage != null)
                        ElevatedButton.icon(
                          onPressed: _uploadImageAndValidate,
                          icon: const Icon(Icons.analytics_outlined),
                          label: const Text("Analyze Now"),
                          style: ElevatedButton.styleFrom(
                            minimumSize: const Size(260, 50),
                            backgroundColor: Colors.green[600],
                            foregroundColor: Colors.white,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(30),
                            ),
                          ),
                        ),
                      const SizedBox(height: 12),
                      if (_heatmapBase64 != null)
                        ElevatedButton.icon(
                          onPressed: _showHeatmapDialog,
                          icon: const Icon(Icons.wb_sunny_outlined),
                          label: const Text("View Heatmap Again"),
                          style: ElevatedButton.styleFrom(
                            minimumSize: const Size(260, 50),
                            backgroundColor: Colors.teal[600],
                            foregroundColor: Colors.white,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(30),
                            ),
                          ),
                        ),
                      const SizedBox(height: 12),
                      ElevatedButton.icon(
                        onPressed: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (context) => const MapScreen(),
                            ),
                          );
                        },
                        icon: const Icon(Icons.map_outlined),
                        label: const Text("View Analysis Map"),
                        style: ElevatedButton.styleFrom(
                          minimumSize: const Size(260, 50),
                          backgroundColor: Colors.orange[800],
                          foregroundColor: Colors.white,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(30),
                          ),
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