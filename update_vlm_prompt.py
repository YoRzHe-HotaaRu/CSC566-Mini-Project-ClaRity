with open('src/config.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Define old and new prompts
old_prompt = '''# Road analysis prompt for VLM
ROAD_ANALYSIS_PROMPT = """
Analyze this aerial satellite image of a road construction site.
Identify which of the 5 road construction layers is shown:

1. Subgrade (in-site soil/backfill) - Earth/soil, brown tones, irregular texture
2. Subbase Course (crushed aggregate) - Coarse stones, gray, rough texture
3. Base Course (crushed aggregate) - Finer aggregate, more uniform gray
4. Binder Course (premix asphalt) - Dark surface with visible aggregate
5. Surface Course (premix asphalt) - Smooth, uniform dark/black surface

Provide your analysis in the following format:
- Layer Number: [1-5]
- Layer Name: [name]
- Confidence: [0-100%]
- Material Observed: [description]
- Texture Characteristics: [description]
- Additional Notes: [any relevant observations]
"""'''

new_prompt = '''# Road analysis prompt for VLM (IMPROVED - Universal prompt)
ROAD_ANALYSIS_PROMPT = """
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

Provide answer in this EXACT format:
LAYER: [1-5]
NAME: [exact name from list above]
CONFIDENCE: [0-100%]
REASONING: [2-3 sentences explaining your choice based on color, texture, and material observed]
"""'''

# Replace
content = content.replace(old_prompt, new_prompt)

# Write back
with open('src/config.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Updated ROAD_ANALYSIS_PROMPT in config.py')
print('New prompt includes:')
print('  - Detailed layer descriptions')
print('  - Analysis checklist')
print('  - Clear output format')
print('  - Works for both close-up and aerial images')
