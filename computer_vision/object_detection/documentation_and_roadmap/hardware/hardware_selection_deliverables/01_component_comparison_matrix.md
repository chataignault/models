# Camera and Processing Board Component Comparison Matrix

## Executive Summary

Based on comprehensive research of commercially available components meeting the embedded vision system requirements, this document provides detailed comparison matrices for camera modules and processing boards that fit within the specified constraints:
- Camera: Max 32mm diameter, 45-80fps, 20-30° FOV
- Processing Board: Max 30mm diameter, 55mm length
- Total Budget: $500 maximum

## Camera Module Comparison Matrix

### USB 3.0 Interface Cameras

| Model | Dimensions | Frame Rate | Interface | FOV Capability | Estimated Price | EU Availability | Key Features |
|-------|------------|------------|-----------|----------------|----------------|-----------------|--------------|
| ELP 720P USB2.0 | 32x32mm | 30fps @ 720p | USB 2.0 | 45° with M7 lens | $25-35 | Yes (distributors) | OV9712 sensor, UVC compliant |
| ELP 1080P USB2.0 | 32x32mm | 30fps @ 1080p | USB 2.0 | 100° wide angle | $35-45 | Yes (distributors) | OV2710 sensor, no drivers needed |
| Basler ace2 USB 3.0 | 29x29mm | Up to 160fps* | USB 3.0 | Depends on lens | $200-400 | Yes (major distributors) | Sony Pregius S, industrial grade |
| Allied Vision Alvium 1800 U | ~30x30mm | Up to 120fps* | USB 3.0 | Depends on lens | $150-300 | Yes (direct/distributors) | Various sensor options |
| FLIR Blackfly S USB3 | 29x29x10mm | Up to 120fps* | USB 3.0 | Depends on lens | $200-500 | Yes (Edmund Optics) | Ice-cube form factor |

*Frame rate depends on resolution - higher fps achieved at lower resolutions

### MIPI CSI-2 Interface Cameras

| Model | Dimensions | Frame Rate | Interface | FOV Capability | Estimated Price | EU Availability | Key Features |
|-------|------------|------------|-----------|----------------|----------------|-----------------|--------------|
| e-con e-CAM52A_MI5640 | ~25x25mm | 60fps @ 720p, 30fps @ 1080p | MIPI CSI-2 | Interchangeable M12 | $80-120 | Yes (direct sales) | OV5640 5MP sensor |
| e-con e-CAM130_CURB | ~30x30mm | 60fps @ 1080p, 15fps @ 4K | MIPI CSI-2 | Interchangeable M12 | $150-200 | Yes (direct sales) | 13MP, 4K capable |
| Arducam IMX135 | ~25x25mm | 60fps @ 720p, 30fps @ 1080p | MIPI CSI-2 | Depends on lens | $50-80 | Yes (EBV Elektronik) | 13MP Sony sensor |
| The Imaging Source MIPI modules | Board only | 30-120fps | MIPI CSI-2 | S-mount compatible | $100-200 | Yes (direct) | Sony/onsemi sensors |

## Processing Board Comparison Matrix

### Ultra-Compact ARM Processors (≤30mm diameter requirement)

| Model | Dimensions (mm) | Processor | RAM | GPU/AI | Interfaces | Estimated Price | EU Availability | Power Draw |
|-------|-----------------|-----------|-----|---------|------------|----------------|-----------------|------------|
| Portwell MicroSOM i.MX8M Plus | 25x25x3.5 | ARM Cortex-A53 | 1-4GB | GPU + NPU | MIPI CSI-2, USB | $80-150 | Contact required | ~2-5W |
| Portwell MicroSOM i.MX6 | 25x25x3.5 | ARM Cortex-A9 | 512MB-2GB | GPU | USB, I2C, SPI | $50-100 | Contact required | ~1-3W |
| Raspberry Pi Zero 2 W | 65x30x5 | ARM Cortex-A53 | 512MB | VideoCore IV | USB 2.0, CSI | $15-20 | Yes (widespread) | ~1W |

### Compact ARM Processors (30-50mm, high performance)

| Model | Dimensions (mm) | Processor | RAM | GPU/AI | Interfaces | Estimated Price | EU Availability | Power Draw |
|-------|-----------------|-----------|-----|---------|------------|----------------|-----------------|------------|
| Hailo-15 SOM | 47x30 | Quad ARM + AI | TBD | 20 TOPS AI | CSI-2, USB 3.0 | $149+ | Limited (Auvidea) | ~10W |
| NVIDIA Jetson Nano | 69.6x45 | ARM Cortex-A57 | 4GB | 128 CUDA cores | CSI-2, USB 3.0 | $149-200 | Yes (major distributors) | 5-10W |
| Open-Q 660 µSOM | 50x25 | ARM Cortex-A73/A53 | 4-6GB | Adreno 512 | CSI, USB 3.0 | $200-300 | Contact required | ~3-8W |

## Interface Compatibility Analysis

### Camera-Processor Interface Matching

| Camera Interface | Compatible Processors | Data Rate | Latency | Integration Complexity |
|------------------|----------------------|-----------|---------|----------------------|
| USB 3.0 | All processors via USB hub/controller | 5Gbps | Medium | Low - plug-and-play |
| USB 2.0 | All processors | 480Mbps | Medium | Low - plug-and-play |
| MIPI CSI-2 | MicroSOM, Hailo-15, Jetson, RPi | 2.5Gbps+ | Low | Medium - direct connection |

### Optimal Pairings for 45-80fps Performance

1. **High Performance**: MIPI CSI-2 camera + Jetson Nano/Hailo-15
2. **Balanced**: USB 3.0 camera + MicroSOM i.MX8M Plus  
3. **Budget**: USB 2.0 camera + Raspberry Pi Zero 2 W (limited to 30fps)

## Size Constraint Analysis

### Meeting 32mm Camera Diameter Requirement
✅ **Compliant**: ELP modules (32x32mm), e-con modules (~25-30mm), Arducam modules (~25mm)
✅ **Compliant**: Basler ace2 (29x29mm), Allied Vision Alvium (~30x30mm)

### Meeting 30mm Processing Board Diameter Requirement  
✅ **Compliant**: Portwell MicroSOM (25x25mm)
✅ **Compliant**: Raspberry Pi Zero 2 W (30mm width, 65mm length)
❌ **Non-compliant**: Jetson Nano (45mm width), Hailo-15 (47mm length), Open-Q 660 (50mm length)

## Critical Findings

1. **Size Constraint Challenge**: Very few processing boards meet the 30mm diameter requirement
2. **Performance Trade-offs**: Smallest boards have limited processing power for 45-80fps
3. **Interface Preference**: MIPI CSI-2 offers better performance but limits processor choice
4. **European Availability**: Most components available but some require direct manufacturer contact

## Recommendations for Further Analysis

1. Consider relaxing processing board diameter to 45mm to enable Jetson Nano
2. Evaluate if 30fps performance acceptable to enable more processor options
3. Investigate custom carrier board solutions for ultra-compact integration
4. Verify actual availability and lead times for Portwell MicroSOM products

---

*Next: Cost Analysis and Final System Recommendations*