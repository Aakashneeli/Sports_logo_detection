# streamlit_app.py
import streamlit as st
import cv2 # Keep for basic types if needed, but direct use minimized here
import tempfile
import asyncio
import os
import pandas as pd
import altair as alt
from pathlib import Path
from src.logo_detector import GeminiLogoDetector, NonRetryableError # Import custom exception
from collections import defaultdict
import traceback
# Function to run async tasks in Streamlit (no changes needed)
def async_run(coroutine):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
             # If loop is running, create a new task in the existing loop
             # This might be necessary if Streamlit uses uvloop or similar
             future = asyncio.ensure_future(coroutine)
             # Create a *new* loop just to run this future to completion
             # This is a workaround for potential nested loop issues.
             temp_loop = asyncio.new_event_loop()
             result = temp_loop.run_until_complete(future)
             temp_loop.close()
             return result
        else:
             # If no loop is running, default behavior
             return loop.run_until_complete(coroutine)
    except RuntimeError:
        # No loop running, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(coroutine)
        # loop.close() # Closing might cause issues if other async things run later
        return result


# Function to get API key (no changes needed)
def get_api_key():
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        # st.sidebar.success("API Key found in environment variables.") # Less verbose
        return api_key
    try:
        if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
            # st.sidebar.success("API Key found in Streamlit secrets.") # Less verbose
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    st.sidebar.warning("API Key not found.")
    api_key = st.sidebar.text_input(
        "Enter your Google Gemini API key:",
        type="password",
        help="Get an API key from Google AI Studio: https://makersuite.google.com/app/apikey"
    )
    if api_key:
        return api_key
    else:
        st.warning("Please provide your Google Gemini API key in the sidebar to proceed.")
        return None


async def run_analysis(detector, video_path, progress_bar):
    """Process video and update progress. Returns True on success, False on failure."""
    try:
        # Reset detector state explicitly before each run
        detector.stats = defaultdict(lambda: {"count": 0, "timestamps": []})
        detector.timeline_data = []
        detector.frame_results_cache = {} # Clear cache too

        await detector.process_video(video_path)
        # Check if any frames were successfully processed, even if no logos found
        success_count = sum(1 for res in detector.frame_results_cache.values() if res['status'] == 'success')
        total_processed = len(detector.frame_results_cache)
        if total_processed == 0: # No frames were even selected
             st.warning("No frames were selected for analysis based on the frame step.")
             progress_bar.progress(1.0, text="Analysis Complete (No frames processed).")
             return True # Technically analysis didn't fail, just did nothing

        if success_count == 0 and total_processed > 0:
             st.error(f"❌ Analysis failed for all {total_processed} selected frames. Check API key and Gemini status.")
             progress_bar.progress(1.0, text="Analysis Failed!")
             return False # Indicate failure

        progress_bar.progress(1.0, text="Analysis Complete!")
        return True # Indicate success
    except NonRetryableError as nre:
        st.error(f"❌ Critical API Error during analysis: {nre}. Please check your API key or quota.")
        progress_bar.progress(1.0, text="Analysis Failed!")
        return False
    except ValueError as ve: # Catch specific errors like file not found
         st.error(f"❌ Input Error: {str(ve)}")
         progress_bar.progress(1.0, text="Analysis Failed!")
         return False
    except Exception as e:
        st.error(f"❌ An unexpected error occurred during analysis: {str(e)}")
        progress_bar.progress(1.0, text="Analysis Failed!")
        # Re-raise or log more details if needed
        print(f"Unhandled analysis error: {e}", exc_info=True) # Log traceback server-side
        return False


def display_analytics(analytics_data):
    """Display analytics: frequency, timeline, co-occurrence, peak viewership."""
    st.subheader("📊 Analytics Results")

    summary = analytics_data.get("summary", {})
    co_occurrence = analytics_data.get("co_occurrence", {})
    peak_viewership = analytics_data.get("peak_viewership", {})
    timeline_data_raw = st.session_state.get('timeline_data_raw', []) # Get raw timeline data from session state

    if not summary and not timeline_data_raw:
        st.info("No logos were detected in the analyzed frames.")
        return

    # --- Tabs for better organization ---
    tab1, tab2, tab3 = st.tabs(["📈 Logo Frequency & Timeline", "🤝 Logo Co-occurrence", "⭐ Peak Viewership"])

    with tab1:
        st.markdown("#### Logo Frequency & Duration")
        if summary:
            freq_data = [
                {
                    'Logo': logo,
                    'Appearances': data['count'],
                    'Est. Duration (s)': data['estimated_total_seconds'],
                    'First Seen (s)': data['first_appearance_sec'],
                    'Last Seen (s)': data['last_appearance_sec'],
                    '% of Detections': data['frequency_pct_overall']
                 }
                for logo, data in summary.items()
            ]
            df_freq = pd.DataFrame(freq_data)
            st.dataframe(df_freq, use_container_width=True)

            st.markdown("#### Logo Appearances (Bar Chart)")
            chart_bar = alt.Chart(df_freq).mark_bar().encode(
                x=alt.X('Appearances:Q', title='Number of Detections (Sampled Frames)'),
                y=alt.Y('Logo:N', title='Logo', sort='-x'),
                tooltip=['Logo', 'Appearances', 'Est. Duration (s)', 'First Seen (s)', 'Last Seen (s)', '% of Detections']
            ).properties(
                title='Total Detections per Logo'
            ).interactive()
            st.altair_chart(chart_bar, use_container_width=True)
        else:
            st.info("No logo frequency data to display.")

        st.markdown("---")
        st.markdown("#### Unique Logos Visible Over Time")
        if timeline_data_raw:
            df_timeline = pd.DataFrame(timeline_data_raw)
            if not df_timeline.empty:
                # Group by timestamp and count unique logos
                df_grouped = df_timeline.groupby('timestamp')['logo'].nunique().reset_index()
                df_grouped.columns = ['Time (seconds)', 'Unique Logos Count']

                chart_line = alt.Chart(df_grouped).mark_line(point=True).encode(
                    x=alt.X('Time (seconds):Q', axis=alt.Axis(format=".1f")), # Format time axis
                    y=alt.Y('Unique Logos Count:Q', title='Number of Unique Logos Detected', scale=alt.Scale(zero=True)), # Ensure Y axis starts at 0
                    tooltip=['Time (seconds)', 'Unique Logos Count']
                ).properties(
                    title='Number of Unique Logos Visible Over Time (Based on Sampled Frames)'
                ).interactive()
                st.altair_chart(chart_line, use_container_width=True)
            else:
                 st.info("No timeline data to display.")
        else:
            st.info("No timeline data available.")


    with tab2:
        st.markdown("#### Top Logo Co-occurrences")
        if co_occurrence:
            co_data = [
                {"Logo Pair": f"{pair[0]} & {pair[1]}", "Count": count}
                for pair, count in co_occurrence.items()
            ]
            df_co = pd.DataFrame(co_data)
            st.dataframe(df_co, use_container_width=True)

            # Optional: Bar chart for co-occurrence
            chart_co_bar = alt.Chart(df_co).mark_bar().encode(
                x=alt.X('Count:Q', title='Number of Times Seen Together'),
                y=alt.Y('Logo Pair:N', title='Logo Pair', sort='-x'),
                tooltip=['Logo Pair', 'Count']
            ).properties(
                title='Most Frequent Logo Pairs in the Same Frame'
            ).interactive()
            st.altair_chart(chart_co_bar, use_container_width=True)
        else:
            st.info("No co-occurrence data available (requires multiple logos detected in the same frame).")

    with tab3:
        st.markdown("#### Peak Logo Viewership")
        if peak_viewership and peak_viewership.get("max_logos", 0) > 0:
            max_logos = peak_viewership["max_logos"]
            timestamps = peak_viewership["timestamps"]
            st.metric(label="Maximum Unique Logos Seen Simultaneously", value=f"{max_logos}")
            st.write(f"This peak occurred at the following time(s) (seconds):")
            # Display timestamps neatly
            ts_str = ", ".join(map(str, timestamps))
            st.code(ts_str) # Use code block for better readability if many timestamps
        else:
            st.info("No peak viewership data available.")


# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="Sports Sponsorship Analytics", page_icon="🏆", layout="wide")

    st.title("🏆 Sports Sponsorship Analytics")
    st.markdown("Upload a sports video to analyze brand logo visibility using Google Gemini.")

    # Get API Key
    api_key = get_api_key()
    if not api_key:
        st.stop()

    # Initialize detector - Use session state to avoid re-initialization on every interaction
    if 'detector' not in st.session_state:
        try:
            st.session_state.detector = GeminiLogoDetector(api_key=api_key)
            print("GeminiLogoDetector initialized.")
        except Exception as e:
            st.error(f"🚨 Error initializing Gemini Detector: {str(e)}")
            st.stop()
    # Ensure API key is updated if it changes (e.g., user inputs it later)
    # This part is tricky, might require re-init if key changes post-init.
    # For simplicity, assume key is stable once provided. Consider adding logic
    # to reinstantiate if the key retrieved differs from the one used in init.

    detector = st.session_state.detector

    # File Uploader
    uploaded_file = st.file_uploader("📤 Upload Video File", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_file:
        st.markdown("---")
        st.subheader("⏳ Processing Video")

        # Use a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tfile:
            tfile.write(uploaded_file.read())
            temp_video_path = tfile.name

        # Clear previous results from session state if a new file is uploaded
        if st.session_state.get('last_uploaded_filename') != uploaded_file.name:
             st.session_state.analysis_done = False
             st.session_state.analytics_data = {}
             st.session_state.timeline_data_raw = []
             st.session_state.processed_segment_bytes = None
             st.session_state.last_uploaded_filename = uploaded_file.name
             print(f"New file uploaded: {uploaded_file.name}. Clearing previous state.")


        analysis_succeeded = False
        processed_segment_bytes = st.session_state.get('processed_segment_bytes') # Try to reuse if already generated

        # Only run analysis if it hasn't been done for this file yet
        if not st.session_state.get('analysis_done'):
            progress_text = "Starting analysis..."
            progress_bar = st.progress(0, text=progress_text)

            try:
                # Run the main video analysis
                progress_bar.progress(0.1, text="Analyzing video frames with Gemini AI...")
                analysis_succeeded = async_run(run_analysis(detector, temp_video_path, progress_bar))
                st.session_state.analysis_done = True # Mark analysis attempt as done

                if analysis_succeeded:
                    # Get analytics and timeline data AFTER successful analysis
                    st.session_state.analytics_data = detector.get_analytics()
                    st.session_state.timeline_data_raw = detector.get_timeline_data() # Store raw data
                    unique_brands_found = len(st.session_state.analytics_data.get("summary", {}))
                    st.success(f"✅ Analysis complete! Found {unique_brands_found} unique brands.")
                    st.session_state.analysis_succeeded = True # Mark as succeeded
                else:
                    st.warning("Analysis finished, but errors occurred or no frames were successfully processed.")
                    st.session_state.analysis_succeeded = False

            except Exception as main_err:
                 st.error(f"An critical error occurred during the analysis process: {main_err}")
                 print(f"Unhandled error in main execution block: {main_err}", exc_info=True) # Log traceback
                 st.session_state.analysis_succeeded = False
                 st.session_state.analysis_done = True # Mark as done even if failed critically

            finally:
                 # Initial cleanup of the uploaded temp video file
                 if Path(temp_video_path).exists():
                     try:
                         os.unlink(temp_video_path)
                     except Exception as e:
                         print(f"Error deleting temporary video file {temp_video_path}: {e}")
        else:
             # Analysis was already done (or attempted) for this file
             print("Analysis results already in session state.")
             analysis_succeeded = st.session_state.get('analysis_succeeded', False)


        # --- Display Results Area ---
        if st.session_state.get('analysis_done'):
            st.markdown("---")

            # 1. Process and Display Video Segment (only if analysis succeeded and logos found)
            st.subheader("🎥 Logo Detection Example (Segment)")
            analytics_summary = st.session_state.get('analytics_data', {}).get('summary', {})
            if analysis_succeeded and analytics_summary:

                # Only generate segment if not already generated and stored
                if not processed_segment_bytes:
                    try:
                        # Find a timestamp where logos were detected
                        first_logo_time = min(
                            (data['first_appearance_sec'] for data in analytics_summary.values() if data.get('first_appearance_sec') is not None),
                             default=0
                        )
                        segment_start_time = max(0, first_logo_time - 1) # Start slightly before

                        with st.spinner("Generating video segment with bounding boxes... (This may take a minute)"):
                             # Use async_run again for the async segment processing
                             # Ensure the original video path is still accessible or passed correctly
                             # Need to re-open the uploaded file if temp_video_path was deleted
                             with open(temp_video_path, 'wb') as f: # Recreate temp file for segment processing
                                 uploaded_file.seek(0) # Go back to the start of the uploaded file buffer
                                 f.write(uploaded_file.read())

                             processed_segment_bytes = async_run(
                                 detector.process_video_segment(temp_video_path, start_time=segment_start_time, duration=7)
                             )
                             st.session_state.processed_segment_bytes = processed_segment_bytes # Store in session state

                             # Clean up the re-created temp file immediately after use
                             if Path(temp_video_path).exists():
                                 try: os.unlink(temp_video_path)
                                 except Exception as e: print(f"Error deleting recreated temp file: {e}")


                        if processed_segment_bytes:
                            st.video(processed_segment_bytes, format="video/avi", start_time=0)
                            
                            # Add a download button for the processed video
                            st.download_button(
                                label="Download Processed Video", 
                                data=processed_segment_bytes,
                                file_name="processed_video.avi",
                                mime="video/avi"
                            )
                        else:
                             st.error("Could not generate the processed video segment. API calls during segment generation might have failed.")
                    except Exception as segment_err:
                        st.error(f"Error generating video segment: {segment_err}")
                        print(f"Segment generation error: {segment_err}") # Print the basic error message to console
                        traceback.print_exc() # Print the full traceback to console
                elif processed_segment_bytes:
                    # Segment already generated, just display it
                     st.video(processed_segment_bytes, format="video/avi", start_time=0)
                     st.caption("Showing previously generated segment.")

            elif not analytics_summary and analysis_succeeded:
                 st.info("Analysis complete, but no logos were detected. Skipping video segment generation.")
            elif not analysis_succeeded:
                 st.warning("Analysis did not complete successfully. Skipping video segment.")


            # 2. Display Analytics (if analysis was attempted)
            display_analytics(st.session_state.get('analytics_data', {}))

        # Cleanup: Ensure temporary files are deleted (moved initial cleanup earlier)
        # No explicit cleanup needed here if bytes are used and temp files deleted promptly

    st.markdown("---")
    st.caption("Powered by Google Gemini ✨ • Built with Streamlit 🚀")

if __name__ == "__main__":
    # Set up asyncio event loop policy for compatibility if needed (esp. on Windows)
    # try:
    #      if asyncio.get_event_loop_policy().__class__.__name__ == 'WindowsProactorEventLoopPolicy' or os.name == 'nt':
    #           asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    #           print("Set asyncio policy to WindowsSelectorEventLoopPolicy")
    # except AttributeError: # Might not exist on non-windows
    #      pass
    main()