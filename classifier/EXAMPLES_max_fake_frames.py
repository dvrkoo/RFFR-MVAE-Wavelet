"""
Example configurations for using max_fake_frames feature.

Copy the desired configuration section into configs/config.py to use it.
"""

# ============================================================================
# EXAMPLE 1: Train with 90 fake frames (minimal data experiment)
# ============================================================================
"""
protocol = "F2F_All_Fake1"
max_fake_frames = 90

Result:
- Fake frames: 90 (from videos 000-089)
- Real frames: 7,200 (all real videos)
- Fake:Real ratio: 1:80
- WandB run name: rffr_2branch_F2F_All_Fake1_90f_2024-12-15_abc123
- WandB tag: "90frames"
"""

# ============================================================================
# EXAMPLE 2: Train with 180 fake frames
# ============================================================================
"""
protocol = "F2F_All_Fake1"
max_fake_frames = 180

Result:
- Fake frames: 180 (from videos 000-179)
- Real frames: 7,200
- Fake:Real ratio: 1:40
- WandB run name: rffr_2branch_F2F_All_Fake1_180f_2024-12-15_abc123
"""

# ============================================================================
# EXAMPLE 3: Train with 360 fake frames
# ============================================================================
"""
protocol = "F2F_All_Fake1"
max_fake_frames = 360

Result:
- Fake frames: 360 (from videos 000-359)
- Real frames: 7,200
- Fake:Real ratio: 1:20
- WandB run name: rffr_2branch_F2F_All_Fake1_360f_2024-12-15_abc123
"""

# ============================================================================
# EXAMPLE 4: Use all frames (default behavior)
# ============================================================================
"""
protocol = "F2F_All_Fake1"
max_fake_frames = None  # or just leave it commented out

Result:
- Fake frames: 720 (all videos)
- Real frames: 7,200
- Fake:Real ratio: 1:10
- WandB run name: rffr_2branch_F2F_All_Fake1_2024-12-15_abc123
"""

# ============================================================================
# EXAMPLE 5: Train with multiple forgeries (mixed mode)
# ============================================================================
"""
protocol = "Mixed_All_Fake1"  # Contains DF, F2F, FSW, NT, FS
max_fake_frames = 450  # Limit total fake frames across all forgeries

Result:
- Fake frames: 450 (first 450 frames from mixed pool)
  - This corresponds to first 90 videos × 5 forgeries
- Real frames: 7,200
- Fake:Real ratio: 1:16
- WandB run name: rffr_2branch_Mixed_All_Fake1_450f_2024-12-15_abc123
"""

# ============================================================================
# EXAMPLE 6: Data scaling experiment (compare different scales)
# ============================================================================
"""
Run multiple experiments to study data efficiency:

Experiment 1: max_fake_frames = 45   # 1:160 ratio
Experiment 2: max_fake_frames = 90   # 1:80 ratio
Experiment 3: max_fake_frames = 180  # 1:40 ratio
Experiment 4: max_fake_frames = 360  # 1:20 ratio
Experiment 5: max_fake_frames = 720  # 1:10 ratio (full fake1)
Experiment 6: max_fake_frames = None # Same as 720 for fake1

All experiments will:
- Use the same 7,200 real frames
- Have unique WandB run names with frame counts
- Be tagged with frame counts for easy filtering
- Log actual frame counts in WandB config
"""

# ============================================================================
# EXAMPLE 7: Using with different fake protocols
# ============================================================================
"""
Works with any protocol:

# Face2Face only
protocol = "F2F_All_Fake1"
max_fake_frames = 90

# Deepfakes only
protocol = "DF_All_Fake1"
max_fake_frames = 90

# FaceSwap only
protocol = "FSW_All_Fake1"
max_fake_frames = 90

# All forgeries mixed
protocol = "Mixed_All_Fake1"
max_fake_frames = 450  # 90 videos × 5 forgeries

Note: For fake3, fake5, etc., max_fake_frames limits total frames, not videos:
protocol = "F2F_All_Fake3"  # 3 frames per video
max_fake_frames = 270  # Gets 270 frames (= 90 videos × 3 frames)
"""

# ============================================================================
# QUICK START GUIDE
# ============================================================================
"""
1. Edit configs/config.py:
   
   # Find the line with max_fake_frames (around line 82)
   max_fake_frames = 90  # Change to 90, 180, 360, or None

2. Verify your protocol is set correctly:
   
   protocol = "F2F_All_Fake1"  # or your desired protocol

3. Run training:
   
   cd /home/nick/GitHub/RFFR/rffr_classifier
   python train.py

4. Check the output - you should see:
   
   *** FAKE FRAME LIMITING ENABLED ***
   Limiting fake data to first 90 frames
   Fake frames: 720 → 90 (videos 000-089)
   Real frames: 7200 (unchanged)
   Fake:Real ratio: 1:80.0

5. Check WandB dashboard:
   
   - Run name includes frame count: model_protocol_90f_timestamp
   - Tags include: "90frames"
   - Config shows: max_fake_frames: 90
   - Config shows: actual_fake_frames: 90
   - Config shows: fake_real_ratio: "1:80.0"
"""

# ============================================================================
# TROUBLESHOOTING
# ============================================================================
"""
Q: I set max_fake_frames = 90 but still see 720 frames?
A: Check that you saved config.py and restarted training. The change should
   print "*** FAKE FRAME LIMITING ENABLED ***" in the output.

Q: Can I limit real frames too?
A: Currently no, only fake frames are limited. Real frames always = 7,200.
   This is by design to maintain robust training on real data.

Q: Does this work with validation/test sets?
A: No, validation and test sets are unchanged. This only affects training.

Q: What if max_fake_frames > available frames?
A: It will use all available frames and print a note. No error.

Q: Can I use this with anomaly_detection_mode?
A: Yes, works with all training modes and configurations.

Q: Will this speed up training?
A: The data loading will be slightly faster with fewer fake samples, but
   the main benefit is for data efficiency experiments, not speed.
"""

# ============================================================================
# BEST PRACTICES
# ============================================================================
"""
1. Always keep max_fake_frames as a multiple of 10 for clean video counts
   (90, 180, 360, 720) in fake1 protocol.

2. Use the same max_fake_frames across multiple runs when comparing
   different architectures or hyperparameters.

3. Add the frame count to your WandB project notes for easy reference.

4. When doing data scaling experiments, run from smallest to largest:
   90 → 180 → 360 → 720 to quickly identify if more data helps.

5. Keep validation/test sets at full size for fair comparison across all
   data scaling experiments.

6. Document your findings in WandB by adding notes to runs explaining
   what you learned about data efficiency at different scales.
"""

print("This file contains example configurations for max_fake_frames.")
print("Copy the desired configuration into configs/config.py to use it.")
print("\nFor full documentation, see: MAX_FAKE_FRAMES_IMPLEMENTATION.md")
