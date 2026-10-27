import azure.cognitiveservices.speech as speechsdk
import json
import os

SPEECH_KEY = "YOUR_AZURE_SPEECH_KEY"
SPEECH_REGION = "eastus"

OUTPUT_FILE = "azure_blendshapes.json"
FPS = 60.0  # Azure animation rate

# ------------------------------------------------------------
# BLENDSHAPE MAPPING (ARKit Standard)
# ------------------------------------------------------------
AZURE_BLENDSHAPE_ORDER = [
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft",
    "eyeSquintLeft", "eyeWideLeft", "eyeBlinkRight", "eyeLookDownRight", "eyeLookInRight",
    "eyeLookOutRight", "eyeLookUpRight", "eyeSquintRight", "eyeWideRight", "jawForward",
    "jawLeft", "jawRight", "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft",
    "mouthRight", "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft", "mouthStretchRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthPressLeft", "mouthPressRight",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthUpperUpLeft", "mouthUpperUpRight", "browDownLeft",
    "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
    "cheekSquintLeft", "cheekSquintRight", "noseSneerLeft", "noseSneerRight", "tongueOut",
    "headRoll", "leftEyeRoll", "rightEyeRoll"
]

def azure_frame_to_named_blendshapes(values):
    if len(values) != 55: return {} 
    return {AZURE_BLENDSHAPE_ORDER[i]: float(values[i]) for i in range(55)}

# ------------------------------------------------------------
# SPEECH CONFIG
# ------------------------------------------------------------
speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)

# Use Jenny or Ava (Ava requires 'FacialExpression' to work)
speech_config.speech_synthesis_voice_name = "en-US-AvaNeural" 

# Use Riff for high compatibility
speech_config.set_speech_synthesis_output_format(
    speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
)

synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

# ------------------------------------------------------------
# VISEME HANDLER
# ------------------------------------------------------------
blendshape_data = []

def on_viseme(evt):
    # If using type="FacialExpression", the animation string is populated
    if evt.animation == "":
        print(f"⚠️ Standard Viseme ID: {evt.viseme_id} (No 3D Data)")
    else:
        print(f"🔥 3D Viseme Received! ({len(evt.animation)} bytes)")
        
        anim = json.loads(evt.animation)
        base_time = evt.audio_offset / 10000000 

        for i, frame in enumerate(anim["BlendShapes"]):
            blendshape_data.append({
                "time": base_time + (i / FPS),
                "blendshapes": azure_frame_to_named_blendshapes(frame)
            })

synthesizer.viseme_received.connect(on_viseme)

# ------------------------------------------------------------
# SSML (FIXED)
# ------------------------------------------------------------
# 🔴 KEY FIX: type="FacialExpression"
ssml = """
<speak version="1.0" 
       xmlns="http://www.w3.org/2001/10/synthesis" 
       xmlns:mstts="http://www.w3.org/2001/mstts" 
       xml:lang="en-US">
  <voice name="en-US-AvaNeural">
    <mstts:viseme type="FacialExpression"/>
    Rainbow has seven colors: Red, orange, yellow, green, blue, indigo, and violet.
  </voice>
</speak>
"""

# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------
print(f"Connecting to {SPEECH_REGION}...")
result = synthesizer.speak_ssml_async(ssml).get()

if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
    print(f"\n🎉 Saved {len(blendshape_data)} frames to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(blendshape_data, f, indent=2)
else:
    print(f"Synthesis failed: {result.reason}")