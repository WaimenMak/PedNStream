# Video Generation Feature

This document describes the new video generation functionality added to the Network Dashboard.

## Overview

The video generation feature allows you to create MP4 or GIF animations of your pedestrian simulation results projected onto real street networks using OpenStreetMap data. This creates compelling demonstration videos showing how pedestrian traffic flows through actual streets.

## Features

- **Real Street Context**: Animations show simulation results on actual OpenStreetMap street networks
- **Multiple Formats**: Support for both MP4 (high quality) and GIF (web-friendly) formats
- **Customizable Settings**: Control video resolution, frame rate, time range, and visualization variables
- **Progress Tracking**: Real-time progress updates during video generation
- **Download Integration**: Direct download links within the Streamlit interface

## Installation

### Required Dependencies

Install the additional dependencies needed for video generation:

```bash
pip install -r video_requirements.txt
```

Or install individually:
```bash
pip install selenium opencv-python webdriver-manager imageio
```

### Chrome/Chromium Browser

The video generation uses Selenium with Chrome WebDriver. Make sure you have Chrome or Chromium installed on your system. The WebDriver will be automatically downloaded and managed.

## Usage

### Via Streamlit Dashboard

1. Run the dashboard as usual:
   ```bash
   streamlit run network_dashboard.py -- --name delft_directions --pos data/delft/node_positions.json
   ```

2. Use the interactive controls to explore your simulation

3. Scroll down to the "🎬 Generate Animation Video" section

4. Configure your video settings:
   - **Time Range**: Select start and end time steps
   - **Format**: Choose MP4 or GIF
   - **FPS**: Set frames per second (1-10)
   - **Resolution**: Set video width and height (in Advanced Options)

5. Click "🎬 Generate Video" and wait for processing

6. Download the generated video using the provided download button

### Programmatic Usage

You can also generate videos programmatically:

```python
from network_dashboard import NetworkDashboard
import json

# Load node positions
with open('data/delft/node_positions.json', 'r') as f:
    pos = {str(k): tuple(v) for k, v in json.load(f).items()}

# Initialize dashboard
dashboard = NetworkDashboard('outputs/delft_directions', pos)

# Generate video
video_path = dashboard.generate_video(
    start_time=0,
    end_time=100,
    variable='density',
    fps=3,
    format='mp4',
    width=1200,
    height=800
)

print(f"Video saved to: {video_path}")
```

## Technical Details

### How It Works

1. **Frame Generation**: For each time step, the system:
   - Creates a Folium map with OpenStreetMap tiles as the base layer
   - Overlays the simulation data (pedestrian density, speed, etc.) on the network links
   - Adds a time step indicator
   - Captures a screenshot using Selenium WebDriver

2. **Video Compilation**: The captured frames are combined into a video using OpenCV:
   - MP4 files use the 'mp4v' codec
   - GIF files use imageio for better quality (falls back to basic method if not available)

3. **Cleanup**: Temporary files are automatically cleaned up after processing

### Performance Considerations

- **Generation Time**: Expect 1-3 seconds per frame depending on network complexity and system performance
- **File Sizes**: 
  - MP4: ~1-5 MB per minute of video
  - GIF: ~2-10 MB per minute of video (varies greatly with content)
- **Memory Usage**: Peak memory usage scales with video resolution and length

### Limitations

- Requires a display server (may need virtual display on headless servers)
- Chrome/Chromium browser must be available
- Large time ranges may take considerable time to process
- Very high resolutions may cause memory issues

## Testing

Test the video generation functionality:

```bash
python test_video_generation.py
```

This will verify that:
- All dependencies are installed correctly
- Required data files are available
- Video generation works with a small test case

## Troubleshooting

### Common Issues

1. **"selenium not found" error**:
   - Install selenium: `pip install selenium`

2. **"ChromeDriver not found" error**:
   - Install webdriver-manager: `pip install webdriver-manager`
   - The Chrome driver will be automatically downloaded

3. **"Display not found" error** (on headless servers):
   - Install virtual display: `sudo apt-get install xvfb`
   - Run with virtual display: `xvfb-run -a python your_script.py`

4. **Poor GIF quality**:
   - Install imageio for better GIF compression: `pip install imageio`

5. **Video generation hangs**:
   - Check that Chrome browser is properly installed
   - Try reducing video resolution
   - Ensure sufficient disk space in temp directory

### Debug Mode

For debugging, you can modify the `_setup_webdriver()` method to run Chrome in non-headless mode by commenting out the `--headless` option.

## File Structure

Generated videos are saved to the `video_outputs/` directory with filenames in the format:
```
network_animation_{variable}_{start_time}to{end_time}_{timestamp}.{format}
```

Example: `network_animation_density_0to100_1642123456.mp4`

## Future Enhancements

Potential improvements for future versions:
- Support for other map styles (satellite, terrain)
- Custom color schemes and styling
- Video watermarks and titles
- Batch processing for multiple simulations
- Integration with cloud storage services
