# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the xxx Embedded CV Challenge project - developing a computer vision module for a camera-guided micro flying device system designed to defend against drones. The project combines hardware selection with computer vision software development.

## Key Files and Structure

### Hardware (Complete - First Draft Done)
- `hardware/requirements.md` - Core hardware constraints and specifications
- `hardware/Hardware_Selection_PRD.md` - Product requirements document
- `hardware/hardware_selection_deliverables/` - Detailed component analysis and recommendations
- `hardware/CLAUDE.md` - Hardware-specific guidance

### Software (Empty - To Be Developed)
- `software/` - Computer vision pipeline implementation (currently empty)
- `Embedded CV challenge.pdf` - Complete project specification and tasks

## Project Context

This is a technical interview challenge for developing a complete embedded CV system with:

**Mission**: Camera-guided micro flying device for last-line defense against drones
- Target: 7-inch racing quadcopters (30x30 cm) at 30-350m range
- Closing velocity: up to 150m/s
- Three phases: Pre-Lock-On → Ground Tracking → Flight Tracking

**Critical Constraints**:
- Processing board: 30mm diameter, ≤55mm length
- Camera + PCB: 32mm diameter
- Budget: $1000 prototype / $250 production
- Timeline: 6 months to flyable prototype
- Must ignore clouds/smoke/ground objects, only track flying objects above horizon

**Software Requirements**:
- Classical computer vision approach (no YOLO)
- Real-time detection and tracking
- Must prioritize most centered target
- Runs on selected embedded hardware
- Live demo capability on laptop

## Working with This Repository

### For Hardware Tasks:
1. Reference `hardware/requirements.md` for complete constraints
2. Use existing deliverables in `hardware_selection_deliverables/`
3. All hardware selection work appears complete

### For Software Tasks:
1. Start with the complete project specification in `Embedded CV challenge.pdf`
2. Focus on classical computer vision methods for drone detection/tracking
3. Design for deployment on hardware selected in hardware phase
4. Implement initially on laptop with clear deployment path
5. Create live demo capability
6. Calculate expected performance on target hardware

## Task Breakdown (8 hours total)
1. **Camera Selection (10%)** - COMPLETED in hardware phase
2. **Processing Board Selection (15%)** - COMPLETED in hardware phase  
3. **Development Process & Timeline (15%)** - Create development plan
4. **Drone Detection & Tracking Pipeline (60%)** - Core CV implementation

## Development Standards

### Software Development:
- Use classical computer vision methods (OpenCV, traditional algorithms)
- Avoid deep learning models like YOLO
- Focus on real-time performance suitable for embedded deployment
- Implement motion detection, object tracking, horizon filtering
- Consider lighting conditions, weather resistance
- Optimize for 45-80 fps camera input

### Performance Considerations:
- Target hardware has limited processing power
- Must handle 150m/s closing velocities  
- Real-time processing is critical
- Consider computational complexity vs accuracy trade-offs

### Code Organization:
- Separate detection and tracking modules
- Modular design for three operational phases
- Clean interfaces for hardware communication
- Testing framework for algorithm validation

## Commands to Run

When implementing the software:
- Use `python` for CV pipeline development
- Install OpenCV: `pip install opencv-python`
- For demo: prepare video file processing capability
- Performance testing: measure fps and latency on laptop

## Notes

- This is a defensive security application (anti-drone system)
- Hardware selection phase is complete - focus on software development
- Dataset access requires NDA signing (mentioned in challenge document)
- Final deliverable includes live demo and performance calculations