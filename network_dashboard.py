"""
Network dashboard visualization. Visualize the results of the simulation.
"""


import folium
import folium.plugins
from branca.colormap import LinearColormap
import json
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_folium import st_folium
import argparse
import os
import tempfile
import time
from pathlib import Path
import base64

# Video generation imports
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import cv2

class NetworkDashboard:
    def __init__(self, data_path, pos, zoom_start=14):
        """
        Initialize the dashboard
        
        Args:
            link_data_path: Path to link_data.json
            pos: Dictionary of node positions {node_id: (x, y)}
            zoom_start: Initial zoom level
        """
        self.pos = pos
        self.zoom_start = zoom_start
        
        # Calculate center from node positions
        lats = [pos[1] for pos in self.pos.values()]
        lons = [pos[0] for pos in self.pos.values()]
        self.center = [
            (max(lats) + min(lats)) / 2,
            (max(lons) + min(lons)) / 2
        ]
        
        link_data_path = os.path.join(data_path, "link_data.json")
        network_params_path = os.path.join(data_path, "network_params.json")
        # Load link data
        with open(link_data_path, 'r') as f:
            self.link_data = json.load(f)
        
        with open(network_params_path, 'r') as f:
            self.network_params = json.load(f)
            
        # Get time steps range
        self.max_time = len(next(iter(self.link_data.values()))['density'])
        
        # Initialize the base map
        self.base_map = None
        
    def create_base_map(self):
        """Create the base map with fixed elements"""
        # Calculate bounds
        lats = [pos[1] for pos in self.pos.values()]
        lons = [pos[0] for pos in self.pos.values()]
        
        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)
        lat_padding = lat_range * 0.01  # Increased padding for better visibility
        lon_padding = lon_range * 0.01
        
        # Calculate bounds
        min_bounds = [min(lats) - lat_padding, min(lons) - lon_padding]
        max_bounds = [max(lats) + lat_padding, max(lons) + lon_padding]
        
        # Create map with bounds
        m = folium.Map(
            location=self.center,
            zoom_start=self.zoom_start,
            max_bounds=True,  # Restrict panning to max bounds
            min_zoom=16,      # Restrict maximum zoom out
            max_zoom=18,      # Restrict maximum zoom in
            bounds=[min_bounds, max_bounds]  # Set initial view bounds
        )
        
        # Add nodes (these don't change)
        for node_id, pos in self.pos.items():
            is_origin = int(node_id) in self.network_params['origin_nodes']
            is_destination = int(node_id) in self.network_params['destination_nodes']

            if is_origin and is_destination:
                folium.Marker(
                    location=[pos[1], pos[0]],
                    icon=folium.Icon(icon='flag', prefix='fa', color='red'),
                    popup=f"Node: {node_id}"
                ).add_to(m)
            elif is_origin:
                folium.Marker(
                    location=[pos[1], pos[0]],
                    icon=folium.Icon(icon='map-marker', prefix='fa'),
                    popup=f"Node: {node_id}"
                ).add_to(m)
            elif is_destination:
                folium.Marker(
                    location=[pos[1], pos[0]],
                    icon=folium.Icon(icon='flag', prefix='fa'),
                    popup=f"Node: {node_id}"
                ).add_to(m)
            else:
                folium.CircleMarker(
                    location=[pos[1], pos[0]],
                    radius=3,
                    color='blue',
                    fill=True,
                    fillColor='lightblue',
                    fillOpacity=0.7,
                    popup=f"Node: {node_id}"
                ).add_to(m)
            
        return m
    
    def update_links(self, m, time_step, variable):
        """Update only the links on the map"""
        # Set value range based on variable
        if variable == 'density':
            vmin, vmax = 0, 8
            colors = ['green', 'yellow', 'red']  # Low to high
        elif variable == 'speed':
            vmin, vmax = 0, 1.5
            colors = ['red', 'yellow', 'green']  # Low to high (inverted)
        else:  # num_pedestrians
            vmin, vmax = 0, 100
            colors = ['green', 'yellow', 'red']  # Low to high
        
        # Create colormap
        colormap = LinearColormap(
            colors=colors,
            vmin=vmin,
            vmax=vmax,
            caption=variable.capitalize()
        )
        
        # Create feature group for links
        links_group = folium.FeatureGroup(name="links")
        
        # Keep track of processed links to handle bidirectional links
        processed_pairs = set()
        
        # Draw links
        for link_id, link_info in self.link_data.items():
            u, v = link_id.split('-')
            reverse_id = f"{v}-{u}"
            
            # Check if this is a bidirectional link
            is_bidirectional = reverse_id in self.link_data
            
            # Skip if we've already processed this link pair
            link_pair = tuple(sorted([u, v]))
            if link_pair in processed_pairs:
                continue
            
            value = link_info[variable][time_step]
            
            # If bidirectional, get the value for the reverse direction
            if is_bidirectional:
                reverse_value = self.link_data[reverse_id][variable][time_step]
                if variable == 'speed':
                    # For speed, use the maximum of both directions
                    value = max(value, reverse_value)
                else:
                    # For density and num_pedestrians, sum the values
                    value = value + reverse_value
                processed_pairs.add(link_pair)
            
            start = self.pos[u]
            end = self.pos[v]
            coords = [(start[1], start[0]), (end[1], end[0])]
            
            # Normalize width to be between 1 and 3 for better visualization
            # width = max(1, min(3, value / vmax * 3))
            if variable == 'num_pedestrians':
                width = min(10, value * 0.5)
            elif variable == 'speed':
                width = max(1, min(10, value * 12))
            else: # density
                width = min(20, value * 12)
            
            # Draw the link
            folium.PolyLine(
                coords,
                color=colormap(value),
                weight=width,
                opacity=0.8,
                popup=f"{value:.1f}" + (" (bi)" if is_bidirectional else "")
            ).add_to(links_group)
        
        # Add links group and colormap to map
        links_group.add_to(m)
        colormap.add_to(m)
        
        return m
    
    def _setup_webdriver(self):
        """Setup Chrome webdriver for screenshot capture"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1200,800")
        chrome_options.add_argument("--disable-gpu")
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
    
    def _capture_map_screenshot(self, folium_map, width=1200, height=800):
        """Capture screenshot of a Folium map"""
        driver = None
        try:
            # Save map to temporary HTML file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                folium_map.save(f.name)
                temp_html_path = f.name
            
            # Setup webdriver
            driver = self._setup_webdriver()
            driver.set_window_size(width, height)
            
            # Load the map and wait for it to render
            driver.get(f"file://{temp_html_path}")
            time.sleep(2)  # Wait for map tiles to load
            
            # Take screenshot
            screenshot_path = temp_html_path.replace('.html', '.png')
            driver.save_screenshot(screenshot_path)
            
            # Read screenshot as numpy array
            frame = cv2.imread(screenshot_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Cleanup
            os.unlink(temp_html_path)
            os.unlink(screenshot_path)
            
            return frame
            
        except Exception as e:
            st.error(f"Error capturing screenshot: {str(e)}")
            return None
        finally:
            if driver:
                driver.quit()
    
    def _create_video_from_frames(self, frames, output_path, fps=2, format='mp4'):
        """Create video from list of frames"""
        if not frames:
            raise ValueError("No frames provided")
        
        height, width, layers = frames[0].shape
        
        if format.lower() == 'mp4':
            # Create MP4 video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
        
        elif format.lower() == 'gif':
            # Create GIF using OpenCV (basic implementation)
            # For better GIF quality, consider using imageio or PIL
            frames_bgr = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]
            
            # Save as individual frames then combine (simplified approach)
            temp_dir = tempfile.mkdtemp()
            frame_paths = []
            
            for i, frame in enumerate(frames_bgr):
                frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
            
            # Use imageio if available, otherwise basic approach
            try:
                import imageio
                with imageio.get_writer(output_path, mode='I', duration=1.0/fps) as writer:
                    for frame in frames:
                        writer.append_data(frame)
            except ImportError:
                st.warning("For better GIF quality, install imageio: pip install imageio")
                # Fallback: save first frame as static image
                cv2.imwrite(output_path.replace('.gif', '.png'), frames_bgr[0])
            
            # Cleanup temp files
            for frame_path in frame_paths:
                os.unlink(frame_path)
            os.rmdir(temp_dir)
    
    def generate_video(self, start_time=0, end_time=None, variable='density', 
                      fps=2, format='mp4', width=1200, height=800, 
                      progress_callback=None):
        """
        Generate video animation of the network simulation
        
        Args:
            start_time: Starting time step
            end_time: Ending time step (None for max_time)
            variable: Variable to visualize ('density', 'speed', 'num_pedestrians')
            fps: Frames per second for video
            format: Output format ('mp4' or 'gif')
            width: Video width in pixels
            height: Video height in pixels
            progress_callback: Optional callback function for progress updates
        
        Returns:
            Path to generated video file
        """
        if end_time is None:
            end_time = self.max_time
        
        # Create output directory
        output_dir = Path("video_outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename
        timestamp = int(time.time())
        filename = f"network_animation_{variable}_{start_time}to{end_time}_{timestamp}.{format}"
        output_path = output_dir / filename
        
        frames = []
        total_frames = end_time - start_time
        
        try:
            for i, t in enumerate(range(start_time, end_time)):
                if progress_callback:
                    progress_callback(i + 1, total_frames)
                
                # Create map for this time step
                m = self.create_base_map()
                m = self.update_links(m, t, variable)
                
                # Add time step indicator
                svg_content = f'<svg xmlns="http://www.w3.org/2000/svg" width="200" height="50"><rect width="200" height="50" fill="white" fill-opacity="0.8" stroke="black"/><text x="10" y="30" font-family="Arial" font-size="16" fill="black">Time Step: {t}</text></svg>'
                svg_encoded = base64.b64encode(svg_content.encode()).decode()
                folium.plugins.FloatImage(
                    f"data:image/svg+xml;base64,{svg_encoded}",
                    bottom=5, left=5
                ).add_to(m)
                
                # Capture screenshot
                frame = self._capture_map_screenshot(m, width, height)
                if frame is not None:
                    frames.append(frame)
                else:
                    st.warning(f"Failed to capture frame for time step {t}")
            
            if frames:
                # Create video from frames
                self._create_video_from_frames(frames, str(output_path), fps, format)
                return str(output_path)
            else:
                raise ValueError("No frames were successfully captured")
                
        except Exception as e:
            st.error(f"Error generating video: {str(e)}")
            return None
    
    def run_dashboard(self):
        """Run the Streamlit dashboard"""
        st.title("Pedestrian Network Traffic Evolution")
        
        # Control panel
        col1, col2 = st.columns([3, 1])
        
        with col2:
            # Variable selection dropdown
            variable = st.selectbox(
                "Select Variable",
                ['density', 'speed', 'num_pedestrians'],
                format_func=lambda x: x.capitalize()
            )
        
        with col1:
            # Time step slider
            time_step = st.slider(
                "Time Step",
                0, self.max_time-1,
                value=0
            )
        
        # Create a new map instead of copying
        m = self.create_base_map()
        m = self.update_links(m, time_step, variable)
        
        # Display map
        st_folium(m, height=600, width=None)
        
        # Video generation section
        st.markdown("---")
        st.subheader("🎬 Generate Animation Video")
        

        col3, col4, col5 = st.columns([2, 1, 1])
        
        with col3:
            # Time range for video
            video_start, video_end = st.slider(
                "Video Time Range",
                0, self.max_time-1,
                value=(0, min(50, self.max_time-1)),  # Default to first 50 steps
                help="Select the time range for the video animation"
            )
        
        with col4:
            # Video format selection
            video_format = st.selectbox(
                "Format",
                ['mp4', 'gif'],
                help="MP4 for high quality, GIF for web sharing"
            )
        
        with col5:
            # Video FPS
            fps = st.slider(
                "FPS",
                1, 10,
                value=2,
                help="Frames per second (higher = faster animation)"
            )
        
        # Advanced options in expander
        with st.expander("Advanced Video Options"):
            col6, col7 = st.columns(2)
            with col6:
                video_width = st.number_input("Width (pixels)", min_value=800, max_value=1920, value=1200)
            with col7:
                video_height = st.number_input("Height (pixels)", min_value=600, max_value=1080, value=800)
        
        # Generate video button
        if st.button("🎬 Generate Video", type="primary"):
            if video_end <= video_start:
                st.error("End time must be greater than start time")
            else:
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(current, total):
                    progress = current / total
                    progress_bar.progress(progress)
                    status_text.text(f"Generating frame {current}/{total}...")
                
                with st.spinner("Generating video... This may take a few minutes."):
                    try:
                        video_path = self.generate_video(
                            start_time=video_start,
                            end_time=video_end + 1,  # +1 because range is exclusive
                            variable=variable,
                            fps=fps,
                            format=video_format,
                            width=int(video_width),
                            height=int(video_height),
                            progress_callback=update_progress
                        )
                        
                        if video_path and os.path.exists(video_path):
                            st.success(f"✅ Video generated successfully!")
                            
                            # Provide download link
                            with open(video_path, "rb") as file:
                                st.download_button(
                                    label=f"📥 Download {video_format.upper()} Video",
                                    data=file.read(),
                                    file_name=os.path.basename(video_path),
                                    mime=f"video/{video_format}" if video_format == 'mp4' else "image/gif"
                                )
                            
                            # Show video info
                            file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
                            st.info(f"📊 Video info: {video_end - video_start + 1} frames, {fps} FPS, {file_size:.1f} MB")
                            
                            # Display preview for small GIFs
                            if video_format == 'gif' and file_size < 10:  # Only show GIFs smaller than 10MB
                                st.image(video_path, caption="Video Preview")
                            
                        else:
                            st.error("❌ Failed to generate video. Please check the logs for details.")
                    
                    except Exception as e:
                        st.error(f"❌ Error generating video: {str(e)}")
                    
                    finally:
                        progress_bar.empty()
                        status_text.empty()

def run_visualization(link_data_path, pos):
    """Helper function to run the dashboard"""
    dashboard = NetworkDashboard(link_data_path, pos)
    dashboard.run_dashboard()

if __name__ == "__main__":
    # Command line: streamlit run network_Dashboard.py -- --name delft --pos data/delft/node_positions.json
    parser = argparse.ArgumentParser(description='Network Dashboard Visualization')
    parser.add_argument('--name', type=str, 
                       default="delft_directions",
                       help='Name of the simulation')
    parser.add_argument('--pos', type=str,
                       default="node_positions.json",
                       help='Path to node_positions.json file')
    
    args = parser.parse_args()
    path_to_pos = os.path.join(".", args.pos)
    # Load node positions
    with open(path_to_pos, 'r') as f:
        pos = {str(k): tuple(v) for k, v in json.load(f).items()}
    
    path_to_data = os.path.join(".", "outputs", args.name)
    # Run visualization with parsed arguments
    run_visualization(path_to_data, pos)