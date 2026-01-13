"""
Improved VLM Prompts for Road Surface Layer Analysis
CSC566 Image Processing Mini Project
"""

# =============================================================================
# IMPROVED VLM PROMPTS
# =============================================================================

# Prompt for CLOSE-UP ground-level photos
ROAD_LAYER_CLOSEUP_PROMPT = """
You are analyzing a CLOSE-UP photograph of a road construction layer.
This is NOT an aerial/satellite view - this is a ground-level photo.

Examine the image carefully and identify which of these 5 road construction layers is shown:

1. SUBGRADE (Layer 1)
   - Material: Natural soil, earth, backfill
   - Appearance: Brown/tan colors, irregular texture
   - Texture: Rough, uneven, soil-like with organic matter
   - Visual cues: Plant roots, soil clumps, earthy tones

2. SUBBASE COURSE (Layer 2)
   - Material: Coarse crushed aggregate (stones)
   - Appearance: Gray, large visible stones (2-4cm size)
   - Texture: Very rough, loose stones, high texture variation
   - Visual cues: Individual stones clearly visible, voids between stones

3. BASE COURSE (Layer 3)
   - Material: Finer crushed aggregate
   - Appearance: Uniform gray, smaller stones (0.5-2cm)
   - Texture: Moderately rough but more uniform than subbase
   - Visual cues: Compacted surface, stone sizes consistent

4. BINDER COURSE (Layer 4)
   - Material: Premix asphalt (coarse asphalt mix)
   - Appearance: Dark gray/black with visible aggregate
   - Texture: Asphalt texture with stones visible on surface
   - Visual cues: Black binder with aggregate texture, not smooth

5. SURFACE COURSE / WEARING COURSE (Layer 5)
   - Material: Fine premix asphalt (finished surface)
   - Appearance: Very dark black or dark gray
   - Texture: Smooth, uniform, minimal texture
   - Visual cues: No visible aggregate, polished or smooth finish

IMPORTANT INSTRUCTIONS:
- Look at the COLOR (brown/soil = subgrade, gray = aggregate, black = asphalt)
- Look at the TEXTURE (rough/loose = subbase, moderately rough = base, smooth = surface)
- Look at the MATERIAL (soil/earth = subgrade, visible stones = subbase/base, asphalt = binder/surface)

Provide your analysis in this EXACT format:
LAYER: [number 1-5]
NAME: [layer name from above]
CONFIDENCE: [0-100%]
REASONING: [brief explanation of why you chose this layer based on color, texture, and material]

Example format:
LAYER: 3
NAME: Base Course
CONFIDENCE: 85%
REASONING: The image shows uniform gray color with fine aggregate texture, stones are 1-2cm in size and surface is compacted
"""

# Prompt for AERIAL/SATELLITE views
ROAD_LAYER_AERIAL_PROMPT = """
You are analyzing an AERIAL or SATELLITE image of a road construction site.
This is a top-down view from above.

Identify which road construction layer is visible:

1. SUBGRADE (Layer 1) - Brown/earthy colors, irregular soil texture
2. SUBBASE COURSE (Layer 2) - Gray, coarse stone texture
3. BASE COURSE (Layer 3) - Gray, finer aggregate texture  
4. BINDER COURSE (Layer 4) - Dark gray with visible aggregate
5. SURFACE COURSE (Layer 5) - Uniform dark/black, smooth finish

Provide your analysis in this EXACT format:
LAYER: [number 1-5]
NAME: [layer name]
CONFIDENCE: [0-100%]
REASONING: [brief explanation]
"""

# Universal prompt (works for both)
ROAD_LAYER_UNIVERSAL_PROMPT = """
Analyze this road construction layer image and identify which layer is shown.

THE 5 ROAD LAYERS:

Layer 1 - SUBGRADE
- What: Natural soil/earth layer
- Color: Brown, tan, earthy tones
- Texture: Irregular, rough, soil-like
- Key features: Organic matter, soil clumps, plant material possible

Layer 2 - SUBBASE COURSE  
- What: Coarse crushed aggregate base
- Color: Gray to light gray
- Texture: Very rough, loose stones visible
- Key features: Large stones (2-4cm), high texture variation, voids

Layer 3 - BASE COURSE
- What: Fine crushed aggregate layer
- Color: Uniform gray
- Texture: Moderately rough but uniform
- Key features: Smaller stones (0.5-2cm), compacted surface

Layer 4 - BINDER COURSE
- What: Coarse asphalt mix
- Color: Dark gray/black with visible stones
- Texture: Asphalt with aggregate texture
- Key features: Black binder material, aggregate visible on surface

Layer 5 - SURFACE/WEARING COURSE
- What: Finished asphalt surface
- Color: Very dark black or dark gray  
- Texture: Smooth to slightly textured
- Key features: No visible aggregate, uniform appearance, polished look

ANALYSIS CHECKLIST:
1. What is the DOMINANT COLOR? (brown → soil/gray → stones/black → asphalt)
2. What is the TEXTURE? (soil-like/rough stones/medium stones/asphalt/smooth)
3. Can you see INDIVIDUAL STONES? (yes → subbase/base, no → asphalt/soil)
4. Is it ASPHALT or SOIL/AGGREGATE? (black & binder → asphalt, brown → soil, gray stones → aggregate)

Provide answer in this format:
LAYER: [1-5]
NAME: [exact name from list]
CONFIDENCE: [0-100%]
REASONING: [2-3 sentences explaining your choice based on color, texture, and material]
"""

# =============================================================================
# PROMPT SELECTION GUIDE
# =============================================================================

"""
How to choose the right prompt:

1. ROAD_LAYER_CLOSEUP_PROMPT
   - Use for: Ground-level close-up photos
   - Distance: 0.5 - 5 meters from surface
   - Shows: Texture, material details clearly
   
2. ROAD_LAYER_AERIAL_PROMPT
   - Use for: Drone/satellite/aerial views
   - Distance: 10+ meters above ground
   - Shows: Large area, construction site context
   
3. ROAD_LAYER_UNIVERSAL_PROMPT
   - Use for: Any image type
   - Works best when unsure of image type
   - Most detailed analysis guide

RECOMMENDATION:
Use ROAD_LAYER_UNIVERSAL_PROMPT for best results in most cases!
"""
