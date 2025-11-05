# Computer Vision Algorithm Research for Drone Detection System

## Executive Summary

This research focuses on classical computer vision algorithms suitable for drone detection and tracking in embedded systems, avoiding YOLO-based approaches. The research covers motion detection, background subtraction, optical flow, and tracking algorithms optimized for computational efficiency and real-time performance on resource-constrained hardware.

## Project Requirements Recap

- **Target**: 7-inch racing quadcopters (30x30 cm) at 30-350m range
- **Closing velocity**: Up to 150m/s 
- **Processing constraints**: 30mm diameter, ≤55mm length processing board
- **Performance**: 45-80 fps real-time processing
- **Approach**: Classical computer vision (no YOLO/deep learning)
- **Environment**: Sky background with cloud/smoke rejection

## Algorithm Categories Researched

### 1. Classical Object Detection Algorithms

#### Histogram of Oriented Gradients (HOG)
- **Description**: Feature descriptor for object detection based on gradient orientation histograms
- **Advantages**: Computationally efficient, interpretable features, well-established
- **Embedded Suitability**: Excellent - low computational requirements, real-time capable
- **Application**: Shape-based drone detection, feature extraction for tracking

#### Viola-Jones Algorithm
- **Description**: First real-time face detection algorithm, cascade-based approach
- **Advantages**: Fast detection speed, optimized for real-time applications
- **Embedded Suitability**: Good - designed for real-time performance
- **Application**: Adaptable for drone detection with appropriate training

#### ORB (Oriented FAST and Rotated BRIEF)
- **Description**: Feature detector and descriptor for keypoint matching
- **Advantages**: Rotation invariant, fast computation, ideal for edge devices
- **Embedded Suitability**: Excellent - specifically designed for limited computational power
- **Application**: Feature-based tracking, object identification

### 2. Background Subtraction Algorithms

#### MOG2 (Mixture of Gaussians 2)
- **Description**: Adaptive Gaussian mixture model for background/foreground segmentation
- **Advantages**: 
  - Adaptive to scene changes and illumination variations
  - Automatic gaussian distribution selection per pixel
  - Built-in shadow detection capability
- **Computational Complexity**: Moderate to high
- **Embedded Performance**: Requires parameter tuning for optimal performance
- **Parameters**: 
  - History: 500 frames (default)
  - Variance Threshold: 16 (default)
  - Shadow detection: Available but reduces speed
- **Application**: Dynamic background handling, outdoor environment adaptation

#### GMM (Gaussian Mixture Models)
- **Description**: Statistical background modeling using gaussian distributions
- **Advantages**: High accuracy, precision, recall, and F1 score
- **Disadvantages**: Computationally expensive
- **Embedded Suitability**: Limited - high computational requirements
- **Application**: High-accuracy scenarios where computational resources allow

#### Simple Frame Differencing
- **Description**: Basic motion detection through consecutive frame subtraction
- **Advantages**: 
  - Extremely low computational cost
  - Simple implementation
  - Robust to illumination changes
- **Disadvantages**: Can miss object centers, creating multiple detections
- **Embedded Suitability**: Excellent - minimal resource requirements
- **Application**: Initial motion detection, trigger for more complex algorithms

#### Census Transform-Based Methods
- **Description**: Pixel ranking-based background subtraction
- **Advantages**: Less computationally intensive than MOG2 and KNN
- **Performance**: Faster than MOG2, suitable for real-time applications
- **Embedded Suitability**: Very good - optimized for limited resources
- **Application**: Resource-constrained real-time motion detection

### 3. Optical Flow Algorithms

#### Lucas-Kanade Method
- **Description**: Sparse optical flow calculation for feature points
- **Advantages**:
  - Well-established and reliable
  - Pyramid implementation handles large motions
  - Low computational load for sparse features
- **Performance**: 20ms+ per frame (CPU implementation)
- **Embedded Considerations**: 
  - Best for slow-moving objects (challenge for 150m/s targets)
  - Requires good feature tracking
- **Application**: Feature-based tracking, velocity estimation

#### Farneback Method  
- **Description**: Dense optical flow using polynomial expansion
- **Advantages**:
  - Computes flow for every pixel
  - No explicit feature detection required
  - Good accuracy for dense motion fields
- **Performance**: ~8ms per frame (GPU implementation)
- **Computational Requirements**: Moderate to high
- **Application**: Dense motion analysis, velocity field computation

#### Hardware-Accelerated Optical Flow (NVIDIA)
- **Description**: Dedicated hardware implementation on modern GPUs
- **Performance**: 2-3ms per frame with high accuracy
- **Advantages**: Extremely fast, leaves GPU cores free for other tasks
- **Embedded Applicability**: Limited to platforms with dedicated optical flow hardware
- **Application**: High-performance tracking when hardware available

### 4. Motion Detection and Tracking

#### Kalman Filter
- **Description**: Optimal state estimation for linear systems
- **Advantages**:
  - Low computational requirements
  - Fast convergence
  - Reliable predictions
  - Easy implementation
- **Applications**:
  - Object state estimation (position, velocity)
  - Prediction for high-speed targets
  - Track continuity maintenance
- **Embedded Suitability**: Excellent - designed for real-time systems
- **Hardware Implementation**: FPGA implementations available

#### Blob Detection
- **Description**: Connected component analysis for object identification
- **Advantages**: Simple, fast, effective for well-defined objects
- **Considerations**: Size-based filtering (30x30cm at various distances)
- **Implementation**: Morphological operations, contour detection
- **Embedded Suitability**: Very good - straightforward algorithms

#### Block-wise Frame Differencing
- **Description**: Enhanced frame differencing with spatial blocks
- **Advantages**: 
  - Improved accuracy over simple frame differencing
  - Reduced computation time
  - Real-time capable
- **Performance**: Better object boundary detection
- **Embedded Suitability**: Excellent - optimized for real-time operation

## Computational Efficiency Analysis

### Performance Hierarchy (Fastest to Slowest)
1. **Hardware Optical Flow**: 2-3ms/frame (dedicated hardware)
2. **Simple Frame Differencing**: <1ms/frame (CPU)
3. **Census Transform Methods**: 2-5ms/frame (CPU)
4. **Farneback Optical Flow**: ~8ms/frame (GPU)
5. **MOG2 Background Subtraction**: 10-15ms/frame (CPU, tuned)
6. **Lucas-Kanade Optical Flow**: 20ms+/frame (CPU)
7. **GMM Background Subtraction**: 25ms+/frame (CPU)

### Memory Requirements
- **Simple algorithms** (frame differencing, blob detection): <50MB
- **MOG2**: 100-200MB (depends on history parameter)
- **Optical flow**: 50-150MB (depends on resolution and method)
- **Kalman filtering**: <10MB (minimal state storage)

### Power Consumption Considerations
- **CPU-only algorithms**: Lower power, suitable for battery operation
- **GPU acceleration**: Higher power but significantly faster processing
- **FPGA implementations**: Optimal power efficiency for specific algorithms

## Algorithm Recommendations by Processing Phase

### Phase 1: Pre-Lock-On (Ground-based detection)
**Primary Algorithm**: MOG2 Background Subtraction
- Adaptive to outdoor conditions
- Handles illumination changes
- Shadow detection for ground object rejection

**Alternative**: Census Transform + Simple Frame Differencing
- Lower computational cost
- Sufficient for initial detection
- Better for resource-constrained systems

### Phase 2: Ground Tracking
**Primary Algorithm**: Kalman Filter + Blob Detection
- Reliable state estimation
- Low computational cost
- Predictive tracking for smooth motion

**Supporting Algorithm**: Lucas-Kanade Optical Flow (sparse)
- Feature-based tracking
- Velocity estimation
- Target verification

### Phase 3: Flight Tracking (High-speed)
**Primary Algorithm**: Kalman Filter with High Update Rate
- Essential for 150m/s closing velocity
- Predictive capability crucial
- Real-time performance

**Supporting Algorithm**: Block-wise Frame Differencing
- Fast motion detection
- Minimal latency
- Robust to high-speed motion

## Hardware Platform Considerations

### CPU-based Implementation
- **Advantages**: Lower power, simpler deployment
- **Recommended Algorithms**: Frame differencing, Kalman filter, simple blob detection
- **Performance**: 30-45 fps achievable with optimized code

### GPU Acceleration
- **Advantages**: Significant speedup for complex algorithms
- **Recommended Algorithms**: Farneback optical flow, MOG2 background subtraction
- **Performance**: 60-80+ fps possible with proper optimization
- **Power Trade-off**: Higher consumption but better performance

### FPGA Implementation
- **Advantages**: Optimal power efficiency, customizable hardware
- **Recommended Algorithms**: Kalman filter, simple motion detection
- **Development Effort**: Higher initial investment, excellent production performance

## Algorithm Combination Strategy

### Recommended Pipeline Architecture
1. **Motion Detection Stage**: Census Transform or Simple Frame Differencing
2. **Background Subtraction**: MOG2 (parameter-tuned for embedded performance)
3. **Object Detection**: Size-filtered blob detection with shape analysis
4. **Tracking**: Kalman filter with correlation-based data association
5. **Prediction**: Extended Kalman filter for high-speed intercept calculations

### Performance Optimization Techniques
- **Multi-threading**: Parallel processing of detection and tracking
- **SIMD Instructions**: Vectorized operations for pixel processing  
- **Memory Management**: Efficient buffer management for real-time constraints
- **Fixed-point Arithmetic**: Reduced precision for speed improvements
- **Region of Interest**: Focus processing on likely target areas

## Cloud and Smoke Rejection

### Horizon Line Detection
- **Method**: Edge detection + Hough line transform
- **Purpose**: Reject ground-based objects
- **Computational Cost**: Low - one-time or infrequent calculation

### Shape Analysis
- **Technique**: Aspect ratio, circularity, and motion pattern analysis
- **Target Profile**: Quadcopter shape characteristics
- **Implementation**: Post-blob detection filtering

### Motion Pattern Recognition
- **Method**: Velocity consistency analysis
- **Drone Characteristics**: Controlled, directed movement vs. random cloud motion
- **Implementation**: Kalman filter innovation monitoring

## 2024 Technology Trends Impact

### Hardware Improvements
- **Edge AI Chips**: Specialized vision processing units becoming available
- **Memory Efficiency**: Better algorithms requiring less RAM
- **Power Optimization**: Algorithms designed for battery-powered operation

### Algorithm Developments
- **Hybrid Approaches**: Combining classical CV with lightweight neural components
- **Adaptive Algorithms**: Self-tuning parameters based on environmental conditions
- **Real-time Optimization**: Focus on sub-10ms processing latencies

## Conclusion and Next Steps

The research indicates that a hybrid approach combining multiple classical algorithms provides the best solution for the drone detection requirements:

1. **Primary Detection**: MOG2 or Census Transform background subtraction
2. **Tracking**: Kalman filter with blob detection
3. **High-speed Phase**: Predictive Kalman filtering with frame differencing
4. **Optimization**: Hardware-specific implementations (CPU, GPU, or FPGA)

The classical approach avoids the computational overhead of deep learning while maintaining sufficient accuracy for the target application. Performance optimization through hardware selection and algorithm tuning will be critical for meeting the 45-80 fps requirement on embedded platforms.

**Recommended Development Priority**:
1. Implement simple frame differencing baseline
2. Add MOG2 background subtraction with parameter tuning
3. Integrate Kalman filter tracking
4. Optimize for target hardware platform
5. Add specialized algorithms for high-speed tracking phase

This algorithm foundation provides a solid starting point for the prototype development phase while maintaining clear paths for embedded optimization.