# src/logo_detector.py
import cv2
import numpy as np
import os
from dotenv import load_dotenv
import google.generativeai as genai
from collections import defaultdict, Counter
from PIL import Image
import asyncio
import time
import random
import hashlib
from typing import Tuple, List, Dict, Any, Optional
import tempfile
import itertools # For co-occurrence

# Define a custom exception for non-retryable API errors
class NonRetryableError(Exception):
    pass

class GeminiLogoDetector:
    def __init__(self, api_key: str = None, frame_step: int = 30):
        if not api_key:
            load_dotenv()
            api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError("API key missing. Set GOOGLE_API_KEY in .env or secrets, or provide it.")

        genai.configure(api_key=api_key)
        # Consider adding safety_settings if needed, be mindful of blocking legitimate content
        # generation_config = genai.types.GenerationConfig(temperature=0.1) # Lower temp for consistency
        self.model = genai.GenerativeModel(
            'gemini-1.5-flash',
             # generation_config=generation_config,
             # safety_settings={'HARASSMENT':'block_none'} # Use cautiously
             )

        self.frame_step_initial = frame_step
        self.frame_step = frame_step # Will be adjusted
        self.stats = defaultdict(lambda: {"count": 0, "timestamps": []})
        self.current_frame_index = 0
        self.frame_rate = 30
        self.max_retries = 5
        self.retry_delay = 2 # Base delay in seconds
        self.use_parallel = True # Keep parallel processing
        self.timeline_data = []  # Raw data: {"logo": logo, "timestamp": timestamp}
        self.frame_results_cache = {} # Optional: Cache frame results if reprocessing needed

    async def _call_gemini_with_retry(self, prompt: str, image: Image.Image, frame_desc: str) -> Optional[str]:
        """Internal helper to call Gemini API with retry logic."""
        attempt = 0
        while attempt <= self.max_retries:
            try:
                # Use generate_content_async for non-blocking calls
                response = await self.model.generate_content_async(
                    [prompt, image],
                    # request_options={"timeout": 120} # Optional: Increase timeout
                )

                # Robust check for response content and blocked prompts
                response_text = None
                if response.parts:
                    response_text = response.text.strip()
                elif response.prompt_feedback and response.prompt_feedback.block_reason:
                    # Blocked prompts are non-retryable failures for this frame
                    print(f"Warning: Prompt for {frame_desc} blocked. Reason: {response.prompt_feedback.block_reason}")
                    raise NonRetryableError(f"Prompt blocked: {response.prompt_feedback.block_reason}")

                # print(f"{frame_desc} - Gemini Response Raw: '{response_text}'") # More detailed debug

                # Handle cases where Gemini might return an empty response successfully
                if response_text is None:
                    print(f"Warning: Received None or empty response part for {frame_desc}. Treating as no detection.")
                    return None # Treat as no logos found

                return response_text

            except NonRetryableError as nre:
                print(f"Non-retryable error for {frame_desc}: {nre}")
                return None # Return None, indicating processing failed for this frame

            except Exception as e:
                err_msg = str(e)
                # Check if Gemini API key is invalid (often contains "API key not valid")
                if "api key not valid" in err_msg.lower():
                     print(f"Fatal Error: Invalid Google API Key detected for {frame_desc}. Stopping retries.")
                     raise NonRetryableError("Invalid API Key") from e

                # Check for specific retryable gRPC/API errors
                is_retryable = (
                    "429" in err_msg or                   # Rate limit
                    "503" in err_msg or                   # Service unavailable
                    "resource exhausted" in err_msg.lower() or # Quota
                    "deadline exceeded" in err_msg.lower() or # Timeout/network issue
                    "internal error" in err_msg.lower()     # Server-side transient issue
                )

                print(f"Error processing {frame_desc} (Attempt {attempt+1}/{self.max_retries+1}): {err_msg}")

                if is_retryable:
                    attempt += 1
                    if attempt <= self.max_retries:
                        # Exponential backoff with jitter
                        sleep_time = self.retry_delay * (2 ** (attempt - 1))
                        sleep_time += random.uniform(0, sleep_time * 0.5) # Add jitter
                        print(f"Retryable error detected. Waiting {sleep_time:.2f}s before retry ({attempt}/{self.max_retries})...")
                        await asyncio.sleep(sleep_time) # Use asyncio.sleep for async context
                    else:
                        print(f"Max retries reached for {frame_desc}. Skipping.")
                        return None # Max retries exceeded
                else:
                    # For other errors (e.g., network issues, unexpected API changes), treat as non-retryable for this frame
                    print(f"Non-retryable error encountered for {frame_desc}. Skipping frame.")
                    raise NonRetryableError(f"Unhandled Exception: {err_msg}") from e

        # Fallback if loop finishes unexpectedly (should technically be caught by retry logic)
        print(f"Warning: Exited retry loop unexpectedly for {frame_desc}. Skipping.")
        return None

    async def process_video(self, video_path: str) -> None:
        """Process the video, detect logos, and update internal stats."""
        print(f"Starting video processing for: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        self.frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration_sec = total_frames / self.frame_rate if self.frame_rate > 0 else 0
        print(f"Video properties: FPS={self.frame_rate:.2f}, Total Frames={total_frames}, Duration={video_duration_sec:.2f}s")

        # Adjust frame_step based on video length
        if video_duration_sec > 300:  # > 5 min
            self.frame_step = max(self.frame_step_initial, int(self.frame_rate * 4))
        elif video_duration_sec > 120: # > 2 min
            self.frame_step = max(self.frame_step_initial, int(self.frame_rate * 2))
        else:
            self.frame_step = max(self.frame_step_initial, int(self.frame_rate))
        print(f"Adjusted frame step to: {self.frame_step} (process 1 frame every ~{self.frame_step/self.frame_rate:.2f} seconds)")

        frames_to_process = []
        frame_indices = []
        actual_frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if actual_frame_count % self.frame_step == 0:
                frames_to_process.append(frame) # No copy needed if used immediately or GIL protects in threads
                frame_indices.append(actual_frame_count)
            actual_frame_count += 1
            self.current_frame_index = actual_frame_count

        cap.release()
        print(f"Extracted {len(frames_to_process)} frames for analysis.")

        if not frames_to_process:
            print("No frames selected for processing.")
            return

        # Reset stats and cache before processing a new video
        self.stats = defaultdict(lambda: {"count": 0, "timestamps": []})
        self.timeline_data = []
        self.frame_results_cache = {}

        start_time = time.time()
        process_func = self.detect_logos # Choose the function to run

        tasks = []
        for i, frame in enumerate(frames_to_process):
            idx = frame_indices[i]
            tasks.append(process_func(frame, idx))

        if self.use_parallel:
            print("Processing frames in parallel...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            print("Processing frames sequentially...")
            results = []
            for task in tasks:
                try:
                    result = await task
                    results.append(result)
                except Exception as e:
                    results.append(e) # Store exception if it occurs

        # Process results
        successful_frames = 0
        failed_frames = 0
        for i, result in enumerate(results):
            idx = frame_indices[i]
            timestamp = idx / self.frame_rate if self.frame_rate > 0 else 0

            if isinstance(result, Exception):
                 # Check if it's our custom NonRetryableError (e.g., invalid key, blocked prompt)
                if isinstance(result, NonRetryableError):
                     print(f"Frame {idx} processing failed permanently: {result}")
                else:
                    print(f"Error processing frame {idx}: {result}") # Other unexpected errors
                failed_frames += 1
                self.frame_results_cache[idx] = {"status": "error", "error": str(result)}
            elif result is not None: # If detect_logos returned a list of logos (can be empty)
                self.frame_results_cache[idx] = {"status": "success", "logos": result}
                if result: # Only update stats if logos were actually found
                    self._update_stats(result, timestamp)
                successful_frames += 1
            else: # Result was None (likely indicates Gemini call failed after retries)
                failed_frames += 1
                self.frame_results_cache[idx] = {"status": "failed_api_call"}


        end_time = time.time()
        print(f"Frame processing finished. Success: {successful_frames}, Failed/Skipped: {failed_frames}. Took {end_time - start_time:.2f} seconds.")


    async def detect_logos(self, frame: np.ndarray, frame_idx: int) -> Optional[List[str]]:
        """Detect logo names in a single frame using Gemini AI. Returns None on unrecoverable error."""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
        except Exception as conversion_err:
            print(f"Error converting frame {frame_idx} to PIL Image: {conversion_err}")
            return [] # Return empty list, as this is a local issue, not an API one

        frame_desc = f"frame {frame_idx} (logo names)"
        prompt = "Identify any brand logos visible in this image. List only the precise brand names, separated by commas. Examples: 'Nike', 'Adidas, Puma', 'Coca-Cola'. If no logos are clearly visible or identifiable, respond with the single word 'None'."

        try:
            logos_text = await self._call_gemini_with_retry(prompt, pil_image, frame_desc)
        except NonRetryableError as e:
             # Propagate the non-retryable error indicator if needed, or handle here
             print(f"Non-retryable error encountered during detect_logos for {frame_desc}: {e}")
             return None # Indicate failure

        if logos_text is None or logos_text.lower() == 'none':
            return [] # No logos detected or API call failed after retries

        # Parse response
        try:
            # Split, strip, convert to lower, filter empty, filter generics
            logos = [
                logo.strip().lower() for logo in logos_text.split(',')
                if logo.strip() and logo.strip().lower() not in ['logo', 'brand', 'image', 'text', 'none']
            ]
            # print(f"Frame {frame_idx} - Detected Logos: {logos}") # Debugging
            return logos
        except Exception as parse_error:
            print(f"Error parsing Gemini response for {frame_desc}: '{logos_text}'. Error: {parse_error}")
            return [] # Return empty list on parsing error

    def _update_stats(self, logos: List[str], timestamp: float) -> None:
        """Update stats for detected logos"""
        processed_logos_in_frame = set()
        for logo in logos:
            if logo and logo not in processed_logos_in_frame:
                self.stats[logo]["count"] += 1
                self.stats[logo]["timestamps"].append(timestamp)
                # Store raw timeline data (can be processed later for co-occurrence etc.)
                self.timeline_data.append({"logo": logo, "timestamp": timestamp})
                processed_logos_in_frame.add(logo)


    def get_analytics(self) -> Dict[str, Any]:
        """Calculate and return analytics summary."""
        if not self.stats:
            return {"summary": {}, "co_occurrence": {}, "peak_viewership": {}}

        analytics_summary = {}
        total_logo_appearances = sum(data["count"] for data in self.stats.values())

        for logo, data in self.stats.items():
            if not logo or data["count"] == 0: continue

            timestamps = sorted(data["timestamps"]) # Ensure sorted
            count = data["count"]
            first_ts = timestamps[0]
            last_ts = timestamps[-1]

            # Estimate duration based on sampled frames
            # Each detection represents roughly frame_step/frame_rate seconds
            estimated_duration_sec = count * (self.frame_step / self.frame_rate) if self.frame_rate > 0 else 0

            # Alternative duration: time between first and last detection (might be misleading if sparse)
            # appearance_span_sec = last_ts - first_ts if count > 1 else (self.frame_step / self.frame_rate)

            analytics_summary[logo] = {
                "count": count,
                "estimated_total_seconds": round(estimated_duration_sec, 2),
                "first_appearance_sec": round(first_ts, 2),
                "last_appearance_sec": round(last_ts, 2),
                "frequency_pct_overall": round(100.0 * count / total_logo_appearances, 1) if total_logo_appearances > 0 else 0,
                # Add raw timestamps if needed elsewhere
                # "timestamps": [round(ts, 2) for ts in timestamps]
            }

        # Sort summary by count descending
        sorted_summary = dict(sorted(analytics_summary.items(), key=lambda item: item[1]['count'], reverse=True))

        # --- Calculate Co-occurrence ---
        co_occurrence = Counter()
        # Group logos by timestamp
        logos_by_time = defaultdict(list)
        for entry in self.timeline_data:
            logos_by_time[entry['timestamp']].append(entry['logo'])

        # Generate pairs for each timestamp with multiple logos
        for timestamp, logos in logos_by_time.items():
            unique_logos = sorted(list(set(logos))) # Get unique logos at this time, sort for consistent pair order
            if len(unique_logos) > 1:
                for pair in itertools.combinations(unique_logos, 2):
                    co_occurrence[pair] += 1

        # Get top N co-occurring pairs
        top_co_occurrences = dict(co_occurrence.most_common(10)) # Get top 10

         # --- Calculate Peak Viewership ---
        peak_info = {"max_logos": 0, "timestamps": []}
        if logos_by_time:
            max_logos = 0
            for timestamp, logos in logos_by_time.items():
                num_unique_logos = len(set(logos))
                if num_unique_logos > max_logos:
                    max_logos = num_unique_logos
                    peak_info["max_logos"] = max_logos
                    peak_info["timestamps"] = [round(timestamp, 2)]
                elif num_unique_logos == max_logos:
                    peak_info["timestamps"].append(round(timestamp, 2))

            # Clean up peak timestamps if too many - maybe show first/last
            if len(peak_info["timestamps"]) > 5:
                 peak_info["timestamps"] = peak_info["timestamps"][:2] + ["..."] + peak_info["timestamps"][-2:]


        return {
            "summary": sorted_summary,
            "co_occurrence": top_co_occurrences,
            "peak_viewership": peak_info
        }

    def get_timeline_data(self) -> List[Dict[str, Any]]:
        """Get raw timeline data for visualization."""
        return sorted(self.timeline_data, key=lambda x: x['timestamp'])

    async def process_video_segment(self, video_path: str, start_time: float = 0, duration: float = 7) -> Optional[bytes]:
        """Process a segment, draw boxes, and return video bytes."""
        print(f"Processing segment: start={start_time:.2f}s, duration={duration:.2f}s")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video for segment processing: {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start_frame = int(start_time * fps)
        if start_frame < 0: start_frame = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        if cap.get(cv2.CAP_PROP_POS_FRAMES) < start_frame - 1: # Check if seeking worked approx
              print(f"Warning: Could not seek accurately to frame {start_frame}. Starting from approx {cap.get(cv2.CAP_PROP_POS_FRAMES)}")


        # Use a temporary file for writing the video segment
        output_filename = f"processed_segment_{int(time.time())}_{random.randint(1000,9999)}.mp4"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Or 'avc1' for H.264
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        print(f"Writing temporary segment to: {output_path}")


        frames_to_process_count = int(duration * fps)
        processed_frame_count = 0
        segment_start_time = time.time()

        while processed_frame_count < frames_to_process_count:
            ret, frame = cap.read()
            if not ret:
                print("Reached end of video or error reading frame during segment processing.")
                break

            current_frame_index_in_segment = start_frame + processed_frame_count

            # --- Detect logos with bounding boxes for *this specific frame* ---
            # This requires an API call per frame - can be slow and costly!
            try:
                 logos_with_boxes = await self.detect_logos_with_boxes(frame, current_frame_index_in_segment)
            except Exception as bbox_err:
                 print(f"Error calling detect_logos_with_boxes for frame {current_frame_index_in_segment}: {bbox_err}")
                 logos_with_boxes = [] # Continue processing segment, but without boxes for this frame

            # Draw bounding boxes
            if logos_with_boxes:
                for logo_name, box in logos_with_boxes:
                    x1, y1, x2, y2 = box
                    # Draw rectangle (Green)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Put text (Logo Name) - ensure it fits
                    label = f"{logo_name}"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    # Adjust text position if near top edge
                    text_y = y1 - 10 if y1 - h - 10 > 0 else y1 + h + 5
                    bg_y_start = text_y - h - 2
                    bg_y_end = text_y + 5
                     # Clamp background rectangle coords to frame boundaries
                    bg_x1 = max(0, x1)
                    bg_y1 = max(0, bg_y_start)
                    bg_x2 = min(frame_width, x1 + w)
                    bg_y2 = min(frame_height, bg_y_end)
                    text_x = x1
                    text_y = min(frame_height - 5, text_y) # Clamp text y


                    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 255, 0), -1) # Filled background
                    cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2) # Black text


            # Write frame to output video
            out.write(frame)
            processed_frame_count += 1

        segment_end_time = time.time()
        print(f"Finished processing segment frames. Took {segment_end_time - segment_start_time:.2f}s. Processed {processed_frame_count} frames.")

        # Release resources BEFORE reading the file
        cap.release()
        out.release()
        # cv2.destroyAllWindows() # Just in case any windows were opened inadvertently

        # --- Read the generated video file into bytes ---
        video_bytes = None
        if os.path.exists(output_path):
             if os.path.getsize(output_path) > 0:
                 try:
                     print(f"Reading generated segment file ({os.path.getsize(output_path)} bytes) into memory...")
                     with open(output_path, 'rb') as f:
                         video_bytes = f.read()
                     print("Successfully read video bytes.")
                 except Exception as read_err:
                     print(f"Error reading segment file {output_path}: {read_err}")
             else:
                 print(f"Error: Output segment file {output_path} is empty.")
             # --- Clean up the temporary file ---
             try:
                 os.unlink(output_path)
                 print(f"Deleted temporary segment file: {output_path}")
             except OSError as e:
                 print(f"Could not delete temporary segment file {output_path}: {e}")
        else:
             print(f"Error: Output segment file {output_path} was not created.")

        return video_bytes # Return bytes or None

    async def detect_logos_with_boxes(self, frame: np.ndarray, frame_idx: int) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """Detect logos and normalized bounding boxes, converting to pixel coordinates. Includes retry logic."""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
        except Exception as conversion_err:
            print(f"Error converting frame {frame_idx} to PIL Image for box detection: {conversion_err}")
            return []

        frame_desc = f"frame {frame_idx} (logo+boxes)"
        prompt = (
            "Identify all brand logos in this image. For each logo found, provide its name and its bounding box coordinates. "
            "Format each detection on a new line like this: 'logo_name: x_min,y_min,x_max,y_max'. "
            "The coordinates MUST be normalized (values between 0.0 and 1.0 relative to image dimensions). "
            "Example: 'nike: 0.1,0.2,0.3,0.4'. If no logos are found, respond ONLY with the word 'None'."
        )

        try:
            response_text = await self._call_gemini_with_retry(prompt, pil_image, frame_desc)
        except NonRetryableError as e:
             print(f"Non-retryable error encountered during detect_logos_with_boxes for {frame_desc}: {e}")
             return [] # Indicate failure for this frame

        if response_text is None or response_text.strip().lower() == 'none':
            # print(f"No logos or failed API call for {frame_desc}") # Debugging
            return []

        logos_with_boxes = []
        lines = response_text.split('\n')
        frame_height, frame_width = frame.shape[:2]

        for line in lines:
            line = line.strip()
            if ':' not in line or ',' not in line:
                # print(f"Skipping malformed line in box detection: '{line}'") # Debugging
                continue

            try:
                parts = line.split(':', 1)
                logo_name = parts[0].strip().lower()
                coords_str = parts[1].strip()

                # Filter out generic/bad names
                if not logo_name or logo_name in ['logo', 'brand', 'image', 'text', 'none']: continue

                coords = list(map(float, coords_str.split(',')))
                if len(coords) != 4:
                    # print(f"Skipping line with incorrect coordinate count: '{line}'") # Debugging
                    continue

                norm_x1, norm_y1, norm_x2, norm_y2 = coords

                # **Crucial:** Clamp normalized coordinates BEFORE potential swapping
                norm_x1 = max(0.0, min(1.0, norm_x1))
                norm_y1 = max(0.0, min(1.0, norm_y1))
                norm_x2 = max(0.0, min(1.0, norm_x2))
                norm_y2 = max(0.0, min(1.0, norm_y2))

                # Convert to pixel coordinates, ensuring x1 < x2 and y1 < y2
                x1 = int(min(norm_x1, norm_x2) * frame_width)
                y1 = int(min(norm_y1, norm_y2) * frame_height)
                x2 = int(max(norm_x1, norm_x2) * frame_width)
                y2 = int(max(norm_y1, norm_y2) * frame_height)

                # Basic sanity check for box size
                min_box_size = 5 # Pixels
                if (x2 - x1) < min_box_size or (y2 - y1) < min_box_size:
                    # print(f"Skipping tiny box for logo '{logo_name}': ({x1},{y1},{x2},{y2})") # Debugging
                    continue

                logos_with_boxes.append((logo_name, (x1, y1, x2, y2)))

            except ValueError as ve:
                print(f"Error parsing coordinates in line for {frame_desc}: '{line}'. Error: {ve}")
                continue
            except Exception as parse_err:
                print(f"Unexpected error parsing line for {frame_desc}: '{line}'. Error: {parse_err}")
                continue

        # print(f"Frame {frame_idx} - Detected Boxes: {logos_with_boxes}") # Debugging
        return logos_with_boxes