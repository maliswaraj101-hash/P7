import os
import azure.cognitiveservices.speech as speechsdk

SPEECH_KEY = "YOUR_AZURE_SPEECH_KEY"

ENDPOINT = "https://eastus.api.cognitive.microsoft.com/"

def real_time_speech_to_text():
    # Load credentials
    speech_config = speechsdk.SpeechConfig(
        subscription = SPEECH_KEY ,
        endpoint = ENDPOINT
    )

    speech_config.speech_recognition_language = "en-US"

    # Use default microphone
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    # --- CALLBACKS ---
    def on_recognizing(evt):
        # Partial (live) results
        print(f"🟡 Partial: {evt.result.text}")

    def on_recognized(evt):
        # Final results
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print(f"✅ Final: {evt.result.text}")
        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            print("❌ No speech recognized")

    def on_canceled(evt):
        print(f"❌ Canceled: {evt.reason}")
        if evt.reason == speechsdk.CancellationReason.Error:
            print(f"Error details: {evt.error_details}")

    def on_session_started(evt):
        print("🎤 Session started")

    def on_session_stopped(evt):
        print("🛑 Session stopped")
        recognizer.stop_continuous_recognition()

    # --- CONNECT EVENTS ---
    recognizer.recognizing.connect(on_recognizing)
    recognizer.recognized.connect(on_recognized)
    recognizer.canceled.connect(on_canceled)
    recognizer.session_started.connect(on_session_started)
    recognizer.session_stopped.connect(on_session_stopped)

    # --- START ---
    print("🎙️ Speak into your microphone (Ctrl+C to stop)...")
    recognizer.start_continuous_recognition()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\nStopping recognition...")
        recognizer.stop_continuous_recognition()

if __name__ == "__main__":
    real_time_speech_to_text()
