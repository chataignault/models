# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a hardware specification repository for the xxx project, focusing on embedded system hardware selection for a flight/drone system with strict physical and budget constraints.

## Key Files and Structure

- `requirements.md` - Core constraint document specifying requirements for camera and processing board components with size limitations and budget constraints

## Project Context

This is a documentation-only repository for hardware procurement decisions. The project involves:

- **Camera Selection**: Requires 20-30° field of view, max 32mm diameter, 45-80 fps capability
- **Processing Board**: Must handle image processing and flight controller communication, max 30mm diameter, 55mm length
- **Budget Constraint**: Total system cost must stay under $500
- **Geographic Preference**: Europe > America > Other regions for sourcing
- **Commercial Availability**: All components must be commercially available

## Working with This Repository

When asked to work with hardware specifications:
1. Reference requirements.md for complete constraint context
2. Consider size constraints (30-32mm diameter limits) as critical requirements  
3. Prioritize cost-effectiveness while meeting technical specifications
4. Verify commercial availability of recommended components
5. Research actual hardware using web search when making recommendations

## Documentation Standards

- Use markdown format for all specifications
- Include detailed justifications for hardware choices based on vendor specs
- Provide vendor information and compatibility analysis
- Structure recommendations with clear sections for components and combinations
- Expected output format includes main items per component, best combinations, and final selection with justification